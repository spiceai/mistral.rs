//! The ring backend's neighbour exchange over RDMA (RoCE v2), used when every link of the
//! ring has an RDMA device on both ends.
//!
//! Each rank keeps two reliable-connected queue pairs: one to its right neighbour, which it
//! only sends on, and one to its left neighbour, which it only receives on. The device for
//! each is the one holding the RoCE v2 address that the kernel routes to that neighbour
//! from, so a ring cabled point to point, where each neighbour sits behind a different port,
//! needs no configuration. libibverbs is loaded at run time, so a host without it, or a
//! ring with any link that cannot use RDMA, keeps the TCP exchange. The handshake runs on
//! every platform, so a rank that cannot use RDMA still answers its neighbours.

use std::{
    collections::VecDeque,
    io::{Read, Write},
    net::{IpAddr, Ipv4Addr, TcpStream, UdpSocket},
    time::{Duration, Instant},
};

use verbs::Link;

/// The most bytes one message carries; a larger payload is sent as several.
const CHUNK: usize = 1 << 20;
/// Receives kept posted on the left link, each with its own `CHUNK` buffer.
const RX_DEPTH: usize = 16;
/// Sends in flight on the right link, each with its own `CHUNK` buffer.
const TX_DEPTH: usize = 2;
/// How long an exchange spins on its completion queues before it starts yielding the CPU.
const SPIN: Duration = Duration::from_micros(50);
/// How often a waiting exchange checks that the left neighbour is still connected, since an
/// RDMA receive never learns that its sender has gone.
const LIVENESS: Duration = Duration::from_millis(100);
/// Set to `1` to keep the ring on TCP.
pub const DISABLE: &str = "MISTRALRS_RING_NO_RDMA";
/// The bytes of a link's endpoint on the wire: present, queue pair number, packet sequence
/// number, GID and path MTU.
const ENDPOINT_LEN: usize = 1 + 4 + 4 + 16 + 4;

pub type Result<T> = std::result::Result<T, String>;

/// What a neighbour needs to connect to one of this rank's queue pairs.
#[derive(Clone, Copy)]
struct Endpoint {
    qpn: u32,
    psn: u32,
    gid: [u8; 16],
    mtu: u32,
}

impl Endpoint {
    fn encode(endpoint: Option<&Self>) -> [u8; ENDPOINT_LEN] {
        let mut out = [0u8; ENDPOINT_LEN];
        if let Some(e) = endpoint {
            out[0] = 1;
            out[1..5].copy_from_slice(&e.qpn.to_le_bytes());
            out[5..9].copy_from_slice(&e.psn.to_le_bytes());
            out[9..25].copy_from_slice(&e.gid);
            out[25..29].copy_from_slice(&e.mtu.to_le_bytes());
        }
        out
    }

    fn decode(bytes: &[u8; ENDPOINT_LEN]) -> Option<Self> {
        let word = |at: usize| {
            u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
        };
        if bytes[0] != 1 {
            return None;
        }
        let mut gid = [0u8; 16];
        gid.copy_from_slice(&bytes[9..25]);
        Some(Self {
            qpn: word(1),
            psn: word(5),
            gid,
            mtu: word(25),
        })
    }
}

/// The local IPv4 address the kernel sends to `peer` from.
fn source_address(peer: IpAddr) -> Result<Ipv4Addr> {
    let socket = UdpSocket::bind("0.0.0.0:0").map_err(|e| e.to_string())?;
    socket
        .connect((peer, 9))
        .map_err(|e| format!("no route to {peer}: {e}"))?;
    match socket.local_addr().map_err(|e| e.to_string())?.ip() {
        IpAddr::V4(ip) => Ok(ip),
        IpAddr::V6(ip) => ip
            .to_ipv4_mapped()
            .ok_or_else(|| format!("{ip} is not IPv4")),
    }
}

/// The ring's two RDMA links: sends go right, receives come from the left.
pub struct RdmaRing {
    right: Link,
    left: Link,
    done: Vec<(usize, usize)>,
    /// Messages received from the left but not yet taken, as `(slot, bytes)` in arrival
    /// order. A neighbour that is ahead sends its next hop before this one is taken.
    arrived: VecDeque<(usize, usize)>,
}

impl std::fmt::Debug for RdmaRing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RdmaRing({})", self.describe())
    }
}

impl RdmaRing {
    /// Connects the ring's links over RDMA. Every rank calls this at the same point, before
    /// any collective, and all of them reach the same answer: the ring, or why it keeps TCP.
    /// A rank that is not `allowed` RDMA still takes part, and its refusal keeps the whole
    /// ring on TCP. An error is a failed handshake over the TCP connections themselves.
    pub fn connect(
        world_size: usize,
        right: &mut TcpStream,
        left: &mut TcpStream,
        allowed: bool,
    ) -> Result<std::result::Result<Self, String>> {
        let peer = |stream: &TcpStream| {
            stream
                .peer_addr()
                .map(|a| a.ip())
                .map_err(|e| e.to_string())
        };
        let link = |stream: &TcpStream, slots: usize, receives: bool| -> Result<Link> {
            if !allowed {
                return Err(format!("{DISABLE} is set"));
            }
            let port = verbs::port_for(source_address(peer(stream)?)?)?;
            Link::new(port, slots, receives)
        };
        let mut right_link = link(right, TX_DEPTH, false);
        let mut left_link = link(left, RX_DEPTH, true);
        // Each side tells each neighbour its end of their shared link.
        let io = |e: std::io::Error| format!("ring handshake: {e}");
        let ours =
            |link: &Result<Link>| Endpoint::encode(link.as_ref().ok().map(Link::endpoint).as_ref());
        right.write_all(&ours(&right_link)).map_err(io)?;
        left.write_all(&ours(&left_link)).map_err(io)?;
        let mut bytes = [0u8; ENDPOINT_LEN];
        right.read_exact(&mut bytes).map_err(io)?;
        let right_remote = Endpoint::decode(&bytes);
        left.read_exact(&mut bytes).map_err(io)?;
        let left_remote = Endpoint::decode(&bytes);
        // Bring both links up where both ends have one.
        let mut verdict: std::result::Result<(), String> = Ok(());
        for (link, remote) in [
            (&mut right_link, right_remote),
            (&mut left_link, left_remote),
        ] {
            let outcome = match (link.as_mut(), remote) {
                (Err(e), _) => Err(e.clone()),
                (Ok(_), None) => {
                    Err("a neighbour has no RDMA device on its link to this rank".to_string())
                }
                (Ok(link), Some(remote)) => link.connect(&remote),
            };
            if verdict.is_ok() {
                verdict = outcome;
            }
        }
        // The ring uses RDMA only if every link can, so every rank passes its own verdict
        // around the ring, and each hop ANDs in the next one.
        let mut all = verdict.is_ok();
        let mut carried = [u8::from(all)];
        for _ in 0..world_size - 1 {
            right.write_all(&carried).map_err(io)?;
            left.read_exact(&mut carried).map_err(io)?;
            all &= carried[0] == 1;
            carried[0] = u8::from(all);
        }
        // Every queue pair is up before the first send, so no message reaches a neighbour
        // that is not yet ready to receive it.
        right.write_all(&[u8::from(all)]).map_err(io)?;
        left.read_exact(&mut carried).map_err(io)?;
        let (Ok(right_link), Ok(left_link), true) = (right_link, left_link, all) else {
            return Ok(Err(verdict.err().unwrap_or_else(|| {
                "another link of the ring cannot use RDMA".to_string()
            })));
        };
        Ok(Ok(Self {
            right: right_link,
            left: left_link,
            done: Vec::new(),
            arrived: VecDeque::with_capacity(RX_DEPTH),
        }))
    }

    /// Which devices the links use, for the log line.
    pub fn describe(&self) -> String {
        format!(
            "sends on {}, receives on {}",
            self.right.describe(),
            self.left.describe()
        )
    }

    /// One hop of the ring: sends all of `send` to the right neighbour while it receives
    /// exactly `recv.len()` bytes from the left one. `left_tcp` is the left neighbour's TCP
    /// connection, idle while the ring runs on RDMA, which is watched for a hang-up.
    pub fn exchange(&mut self, send: &[u8], recv: &mut [u8], left_tcp: &TcpStream) -> Result<()> {
        let outgoing = send.len().div_ceil(CHUNK);
        let incoming = recv.len().div_ceil(CHUNK);
        let (mut posted, mut sent, mut received) = (0, 0, 0);
        let mut idle_since: Option<Instant> = None;
        let mut checked = Instant::now();
        while sent < outgoing || received < incoming {
            let mut progressed = false;
            while posted < outgoing && posted - sent < TX_DEPTH {
                let chunk = &send[posted * CHUNK..send.len().min((posted + 1) * CHUNK)];
                let slot = posted % TX_DEPTH;
                self.right.buf[slot * CHUNK..][..chunk.len()].copy_from_slice(chunk);
                self.right.post_send(slot, chunk.len())?;
                posted += 1;
                progressed = true;
            }
            self.right.completions(&mut self.done)?;
            sent += self.done.len();
            progressed |= !self.done.is_empty();
            self.left.completions(&mut self.done)?;
            self.arrived.extend(self.done.iter().copied());
            while received < incoming {
                let Some((slot, bytes)) = self.arrived.pop_front() else {
                    break;
                };
                progressed = true;
                let at = received * CHUNK;
                let expected = CHUNK.min(recv.len() - at);
                if slot >= RX_DEPTH || bytes != expected {
                    return Err(format!(
                        "ring all-reduce received {bytes} bytes in slot {slot} from the left neighbour where {expected} were due; the ranks disagree on the tensor being reduced"
                    ));
                }
                recv[at..at + bytes].copy_from_slice(&self.left.buf[slot * CHUNK..][..bytes]);
                self.left.post_receive(slot)?;
                received += 1;
            }
            if progressed {
                idle_since = None;
                continue;
            }
            let idle = *idle_since.get_or_insert_with(Instant::now);
            if idle.elapsed() < SPIN {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
            if checked.elapsed() > LIVENESS {
                checked = Instant::now();
                check_left(left_tcp)?;
            }
        }
        Ok(())
    }
}

/// Fails if the left neighbour has hung up its TCP connection.
fn check_left(left_tcp: &TcpStream) -> Result<()> {
    let io = |e: std::io::Error| format!("left neighbour: {e}");
    left_tcp.set_nonblocking(true).map_err(io)?;
    let mut byte = [0u8; 1];
    let peeked = left_tcp.peek(&mut byte);
    left_tcp.set_nonblocking(false).map_err(io)?;
    match peeked {
        Ok(0) => Err("the left neighbour closed its connection".to_string()),
        Ok(_) => Err("the left neighbour sent TCP data while the ring runs on RDMA".to_string()),
        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => Ok(()),
        Err(e) => Err(io(e)),
    }
}

#[cfg(target_os = "linux")]
mod verbs {
    use std::{net::Ipv4Addr, sync::Arc};

    use sideway::ibverbs::{
        address::{AddressHandleAttribute, Gid, GidType},
        completion::{ExtendedCompletionQueue, GenericCompletionQueue, WorkCompletionStatus},
        device::{DeviceInfo, DeviceList},
        device_context::{DeviceContext, Mtu},
        memory_region::MemoryRegion,
        protection_domain::ProtectionDomain,
        queue_pair::{
            ExtendedQueuePair, PostSendGuard, QueuePair, QueuePairAttribute, QueuePairState,
            SetScatterGatherEntry, WorkRequestFlags,
        },
        AccessFlags,
    };

    use super::{Endpoint, Result, CHUNK};

    /// The RDMA device and RoCE v2 address this host reaches a neighbour from.
    pub(super) struct Port {
        ctx: Arc<DeviceContext>,
        name: String,
        port: u8,
        gid_index: u8,
        gid: Gid,
        mtu: Mtu,
    }

    /// The RDMA port holding `local` as a RoCE v2 address.
    pub(super) fn port_for(local: Ipv4Addr) -> Result<Port> {
        let target = local.to_ipv6_mapped().octets();
        let devices = DeviceList::new().map_err(|e| format!("no RDMA devices: {e}"))?;
        for device in devices.iter() {
            let Ok(ctx) = device.open() else { continue };
            let Ok(table) = ctx.query_gid_table() else {
                continue;
            };
            for entry in table {
                if entry.gid_type() != GidType::RoceV2 || entry.gid().raw != target {
                    continue;
                }
                let (Ok(port), Ok(gid_index)) = (
                    u8::try_from(entry.port_num()),
                    u8::try_from(entry.gid_index()),
                ) else {
                    continue;
                };
                let mtu = ctx
                    .query_port(port)
                    .map_err(|e| e.to_string())?
                    .active_mtu();
                return Ok(Port {
                    name: device.name(),
                    ctx,
                    port,
                    gid_index,
                    gid: entry.gid(),
                    mtu,
                });
            }
        }
        Err(format!("no RDMA device holds {local} as a RoCE v2 address"))
    }

    /// One reliable-connected queue pair and its registered buffers. Fields drop in order: the
    /// queue pair before its completion queue, the registration before the memory it covers.
    pub(super) struct Link {
        qp: ExtendedQueuePair,
        cq: Arc<ExtendedCompletionQueue>,
        mr: Arc<MemoryRegion>,
        pub(super) buf: Vec<u8>,
        _pd: Arc<ProtectionDomain>,
        port: Port,
        psn: u32,
    }

    impl Link {
        pub(super) fn new(port: Port, slots: usize, receives: bool) -> Result<Self> {
            let pd = port.ctx.alloc_pd().map_err(|e| e.to_string())?;
            let buf = vec![0u8; slots * CHUNK];
            // SAFETY: `buf` lives in the same `Link` as `mr`, which drops first, and is never
            // reallocated.
            let mr =
                unsafe { pd.reg_mr(buf.as_ptr() as usize, buf.len(), AccessFlags::LocalWrite) }
                    .map_err(|e| e.to_string())?;
            let depth = u32::try_from(slots).map_err(|e| e.to_string())?;
            let cq = port
                .ctx
                .create_cq_builder()
                .setup_cqe(depth + 1)
                .build_ex()
                .map_err(|e| e.to_string())?;
            let handle = GenericCompletionQueue::from(Arc::clone(&cq));
            let mut qp = pd
                .create_qp_builder()
                .setup_send_cq(handle.clone())
                .setup_recv_cq(handle)
                .setup_max_send_wr(if receives { 1 } else { depth })
                .setup_max_recv_wr(if receives { depth } else { 1 })
                .setup_max_send_sge(1)
                .setup_max_recv_sge(1)
                .build_ex()
                .map_err(|e| e.to_string())?;
            let mut attr = QueuePairAttribute::new();
            attr.setup_state(QueuePairState::Init)
                .setup_pkey_index(0)
                .setup_port(port.port)
                .setup_access_flags(AccessFlags::LocalWrite);
            qp.modify(&attr).map_err(|e| e.to_string())?;
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.subsec_nanos());
            let psn = (std::process::id() ^ nanos ^ qp.qp_number()) & 0x00ff_ffff;
            let mut link = Self {
                qp,
                cq,
                mr,
                buf,
                _pd: pd,
                port,
                psn,
            };
            if receives {
                for slot in 0..slots {
                    link.post_receive(slot)?;
                }
            }
            Ok(link)
        }

        /// The device and GID index the link uses, for the log line.
        pub(super) fn describe(&self) -> String {
            format!("{} (GID {})", self.port.name, self.port.gid_index)
        }

        pub(super) fn endpoint(&self) -> Endpoint {
            Endpoint {
                qpn: self.qp.qp_number(),
                psn: self.psn,
                gid: self.port.gid.raw,
                mtu: self.port.mtu as u32,
            }
        }

        /// Brings the queue pair up to send to, and receive from, `remote`.
        pub(super) fn connect(&mut self, remote: &Endpoint) -> Result<()> {
            let mtu = if remote.mtu < self.port.mtu as u32 {
                Mtu::from(remote.mtu)
            } else {
                self.port.mtu
            };
            let gid = Gid { raw: remote.gid };
            let mut ah = AddressHandleAttribute::new();
            ah.setup_port(self.port.port)
                .setup_grh_src_gid_index(self.port.gid_index)
                .setup_grh_dest_gid(&gid)
                .setup_grh_hop_limit(64);
            let mut attr = QueuePairAttribute::new();
            attr.setup_state(QueuePairState::ReadyToReceive)
                .setup_path_mtu(mtu)
                .setup_dest_qp_num(remote.qpn)
                .setup_rq_psn(remote.psn)
                .setup_max_dest_read_atomic(0)
                .setup_min_rnr_timer(1)
                .setup_address_vector(&ah);
            self.qp.modify(&attr).map_err(|e| format!("RTR: {e}"))?;
            let mut attr = QueuePairAttribute::new();
            attr.setup_state(QueuePairState::ReadyToSend)
                .setup_sq_psn(self.psn)
                .setup_timeout(14)
                .setup_retry_cnt(7)
                .setup_rnr_retry(7)
                .setup_max_read_atomic(0);
            self.qp.modify(&attr).map_err(|e| format!("RTS: {e}"))
        }

        fn slot_addr(&self, slot: usize) -> u64 {
            self.buf.as_ptr() as u64 + (slot * CHUNK) as u64
        }

        pub(super) fn post_receive(&mut self, slot: usize) -> Result<()> {
            let (lkey, addr) = (self.mr.lkey(), self.slot_addr(slot));
            let mut guard = self.qp.start_post_recv();
            let wr = guard.construct_wr(slot as u64);
            // SAFETY: the slot is inside the registered buffer.
            unsafe { wr.setup_sge(lkey, addr, CHUNK as u32) };
            guard.post().map_err(|e| e.to_string())
        }

        pub(super) fn post_send(&mut self, slot: usize, len: usize) -> Result<()> {
            let (lkey, addr) = (self.mr.lkey(), self.slot_addr(slot));
            let mut guard = self.qp.start_post_send();
            let wr = guard
                .construct_wr(slot as u64, WorkRequestFlags::Signaled)
                .setup_send();
            // SAFETY: the slot is inside the registered buffer and holds `len` bytes.
            unsafe { wr.setup_sge(lkey, addr, len as u32) };
            guard.post().map_err(|e| e.to_string())
        }

        /// The completions waiting on this link, as `(slot, bytes)`; an error for a failed one.
        pub(super) fn completions(&self, out: &mut Vec<(usize, usize)>) -> Result<()> {
            out.clear();
            let Ok(mut poller) = self.cq.start_poll() else {
                return Ok(());
            };
            for wc in &mut poller {
                if wc.status() != WorkCompletionStatus::Success as u32 {
                    return Err(format!(
                        "RDMA work request {} on {} failed: {:?}",
                        wc.wr_id(),
                        self.port.name,
                        WorkCompletionStatus::from(wc.status())
                    ));
                }
                out.push((wc.wr_id() as usize, wc.byte_len() as usize));
            }
            Ok(())
        }
    }
}

/// Without libibverbs there is no link to open, so every rank of the ring keeps TCP.
#[cfg(not(target_os = "linux"))]
mod verbs {
    use std::{convert::Infallible, net::Ipv4Addr};

    use super::{Endpoint, Result};

    pub(super) enum Port {}

    pub(super) struct Link {
        pub(super) buf: Vec<u8>,
        never: Infallible,
    }

    pub(super) fn port_for(_local: Ipv4Addr) -> Result<Port> {
        Err("RDMA needs Linux".to_string())
    }

    impl Link {
        pub(super) fn new(port: Port, _slots: usize, _receives: bool) -> Result<Self> {
            match port {}
        }

        pub(super) fn describe(&self) -> String {
            match self.never {}
        }

        pub(super) fn endpoint(&self) -> Endpoint {
            match self.never {}
        }

        pub(super) fn connect(&mut self, _remote: &Endpoint) -> Result<()> {
            match self.never {}
        }

        pub(super) fn post_receive(&mut self, _slot: usize) -> Result<()> {
            match self.never {}
        }

        pub(super) fn post_send(&mut self, _slot: usize, _len: usize) -> Result<()> {
            match self.never {}
        }

        pub(super) fn completions(&self, _out: &mut Vec<(usize, usize)>) -> Result<()> {
            match self.never {}
        }
    }
}
