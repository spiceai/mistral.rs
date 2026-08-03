use std::{
    env, fs,
    ops::Range,
    path::{Component, Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering as AtomicOrdering},
        Arc, OnceLock,
    },
    thread,
    time::Duration,
};

use anyhow::{anyhow, Result};
use hf_hub::{
    api::{
        sync::{Api, ApiBuilder, ApiError, ApiRepo},
        tokio::{
            Api as AsyncApi, ApiBuilder as AsyncApiBuilder, ApiError as AsyncApiError,
            ApiRepo as AsyncApiRepo,
        },
    },
    Cache, Repo, RepoType,
};
use signal_hook::consts::TERM_SIGNALS;
use tokio::io::{AsyncReadExt, AsyncSeekExt};
use tracing::{trace, warn};

use crate::utils::tokens::get_token;

/// Env variable that, when set to a truthy value, disables all network calls
/// to the Hugging Face Hub. Only cached files are used.
pub const HF_HUB_OFFLINE_ENV: &str = "HF_HUB_OFFLINE";

/// Returns true when the user has requested fully-offline operation via
/// `HF_HUB_OFFLINE`. Accepted truthy values: `1`, `true`, `yes`, `on`
/// (case-insensitive). Anything else, or unset, is treated as online.
pub fn is_hf_hub_offline() -> bool {
    matches!(
        env::var(HF_HUB_OFFLINE_ENV)
            .ok()
            .map(|v| v.trim().to_ascii_lowercase()),
        Some(ref v) if matches!(v.as_str(), "1" | "true" | "yes" | "on")
    )
}

fn offline_repo(model_id: &Path, revision: &str) -> Repo {
    Repo::with_revision(
        model_id.display().to_string(),
        RepoType::Model,
        revision.to_string(),
    )
}

pub(crate) fn offline_cache_repo(model_id: &Path, revision: &str) -> hf_hub::CacheRepo {
    let cache = hf_hub_cache_dir().map(Cache::new).unwrap_or_default();
    cache.repo(offline_repo(model_id, revision))
}

pub(crate) fn offline_missing_file_error(
    model_id: &Path,
    file: &str,
    revision: &str,
) -> anyhow::Error {
    anyhow!(
        "`{HF_HUB_OFFLINE_ENV}` is set, so Hugging Face was not queried. `{file}` for `{}` (revision `{revision}`) was not found in the local Hugging Face cache. \
         Unset `{HF_HUB_OFFLINE_ENV}` to fetch it online, or pre-download it with `huggingface-cli download {} {file} --revision {revision}`.",
        model_id.display(),
        model_id.display()
    )
}

fn offline_missing_snapshot_error(model_id: &Path, revision: &str) -> anyhow::Error {
    anyhow!(
        "`{HF_HUB_OFFLINE_ENV}` is set, so Hugging Face was not queried. No local Hugging Face snapshot was found for `{}` (revision `{revision}`). \
         Unset `{HF_HUB_OFFLINE_ENV}` to fetch it online, or pre-download it with `huggingface-cli download {} --revision {revision}`.",
        model_id.display(),
        model_id.display()
    )
}

fn offline_snapshot_files(model_id: &Path, revision: &str) -> Vec<String> {
    fn walk(root: &Path, dir: &Path, out: &mut Vec<String>) -> std::io::Result<()> {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                walk(root, &path, out)?;
            } else if let Ok(rel) = path.strip_prefix(root) {
                out.push(rel.to_string_lossy().replace('\\', "/"));
            }
        }
        Ok(())
    }

    let Some(cache_dir) = hf_hub_cache_dir() else {
        return Vec::new();
    };
    let repo = offline_repo(model_id, revision);
    let folder = repo.folder_name();
    let ref_path = cache_dir.join(&folder).join("refs").join(revision);
    // Refs file is typically a branch/tag name; fall back to treating revision as a literal commit.
    let commit = fs::read_to_string(&ref_path)
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| revision.to_string());
    let snapshot_dir = cache_dir.join(&folder).join("snapshots").join(&commit);
    if !snapshot_dir.is_dir() {
        return Vec::new();
    }
    let mut files = Vec::new();
    let _ = walk(&snapshot_dir, &snapshot_dir, &mut files);
    files
}

fn looks_like_local_path(model_id: &Path) -> bool {
    model_id.is_absolute()
        || model_id
            .components()
            .any(|c| matches!(c, Component::CurDir | Component::ParentDir))
        || model_id
            .to_string_lossy()
            .starts_with(['~', std::path::MAIN_SEPARATOR])
}

fn local_resolution_hint(model_id: &Path) -> String {
    if looks_like_local_path(model_id) {
        format!(
            "`{}` looks like a local path, but that path does not exist.",
            model_id.display()
        )
    } else {
        format!(
            "No local directory exists at `{}`, so mistral.rs treated it as a Hugging Face model ID.",
            model_id.display()
        )
    }
}

fn offline_cache_hint(model_id: &Path, revision: &str, file: Option<&str>) -> Option<String> {
    if is_hf_hub_offline() {
        return None;
    }

    if let Some(file) = file {
        if offline_cache_repo(model_id, revision).get(file).is_some() {
            return Some(format!(
                "`{file}` is present in the local Hugging Face cache. If you are offline, set `{HF_HUB_OFFLINE_ENV}=1` to use cached files only."
            ));
        }
        return None;
    }

    if !offline_snapshot_files(model_id, revision).is_empty() {
        return Some(format!(
            "A local Hugging Face snapshot exists for revision `{revision}`. If you are offline, set `{HF_HUB_OFFLINE_ENV}=1` to use cached files only."
        ));
    }

    None
}

#[derive(Clone, Debug)]
pub(crate) struct RemoteAccessIssue {
    pub status_code: Option<u16>,
    pub message: String,
    pub file: Option<String>,
    pub revision: String,
}

/// Resolve the Hugging Face home directory.
///
/// Precedence:
/// 1. HF_HOME
/// 2. ~/.cache/huggingface
pub fn hf_home_dir() -> Option<PathBuf> {
    let dir = env::var("HF_HOME")
        .ok()
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|home| home.join(".cache").join("huggingface")));

    if let Some(ref dir) = dir {
        if let Err(err) = fs::create_dir_all(dir) {
            warn!(
                "Could not create Hugging Face home directory `{}`: {err}",
                dir.display()
            );
        }
    }

    dir
}

/// Resolve the Hugging Face Hub cache directory.
///
/// Precedence:
/// 1. HF_HUB_CACHE
/// 2. HF_HOME/hub
/// 3. ~/.cache/huggingface/hub
pub fn hf_hub_cache_dir() -> Option<PathBuf> {
    let dir = env::var("HF_HUB_CACHE")
        .ok()
        .map(PathBuf::from)
        .or_else(|| hf_home_dir().map(|home| home.join("hub")));

    if let Some(ref dir) = dir {
        if let Err(err) = fs::create_dir_all(dir) {
            warn!(
                "Could not create Hugging Face hub cache directory `{}`: {err}",
                dir.display()
            );
        }
    }

    dir
}

/// Resolve the Hugging Face token file path.
pub fn hf_token_path() -> Option<PathBuf> {
    hf_home_dir().map(|home| home.join("token"))
}

pub(crate) fn build_api(
    token_source: &crate::pipeline::TokenSource,
    progress: bool,
) -> Result<Api> {
    build_api_with_cache(token_source, progress, None)
}

pub(crate) fn build_api_with_cache(
    token_source: &crate::pipeline::TokenSource,
    progress: bool,
    cache: Option<Cache>,
) -> Result<Api> {
    let token = get_token(token_source)?;
    let cache = cache
        .or_else(|| crate::GLOBAL_HF_CACHE.get().cloned())
        .or_else(|| hf_hub_cache_dir().map(Cache::new))
        .unwrap_or_else(Cache::from_env);
    let mut api = ApiBuilder::from_cache(cache)
        .with_progress(progress)
        .with_token(token);
    if let Some(cache_dir) = hf_hub_cache_dir() {
        api = api.with_cache_dir(cache_dir);
    }
    api.build().map_err(Into::into)
}

pub(crate) fn build_async_api(
    token_source: &crate::pipeline::TokenSource,
    progress: bool,
) -> Result<AsyncApi> {
    build_async_api_with_cache(token_source, progress, None)
}

pub(crate) fn build_async_api_with_cache(
    token_source: &crate::pipeline::TokenSource,
    progress: bool,
    cache: Option<Cache>,
) -> Result<AsyncApi> {
    let token = get_token(token_source)?;
    let cache = cache
        .or_else(|| crate::GLOBAL_HF_CACHE.get().cloned())
        .or_else(|| hf_hub_cache_dir().map(Cache::new))
        .unwrap_or_else(Cache::from_env);
    let mut api = AsyncApiBuilder::from_cache(cache)
        .with_progress(progress)
        .with_token(token);
    if let Some(cache_dir) = hf_hub_cache_dir() {
        api = api.with_cache_dir(cache_dir);
    }
    api.build().map_err(Into::into)
}

pub async fn list_model_files(
    model_id: &str,
    revision: &str,
    token_source: &crate::pipeline::TokenSource,
    should_error: bool,
) -> Result<Vec<String>> {
    let api = build_async_api(token_source, false)?;
    let repo = api.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        revision.to_string(),
    ));
    list_repo_files_async(&repo, Path::new(model_id), should_error, revision).await
}

pub fn get_model_file(
    model_id: &str,
    revision: &str,
    file: &str,
    token_source: &crate::pipeline::TokenSource,
) -> Result<PathBuf> {
    let api = build_api(token_source, true)?;
    let repo = Arc::new(api.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        revision.to_string(),
    )));
    get_file(&repo, Path::new(model_id), file, revision)
}

pub async fn try_get_model_file(
    model_id: &str,
    revision: &str,
    file: &str,
    token_source: &crate::pipeline::TokenSource,
) -> Result<Option<PathBuf>> {
    let api = build_async_api(token_source, false)?;
    let repo = api.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        revision.to_string(),
    ));
    try_get_file_async(&repo, Path::new(model_id), file, revision)
        .await
        .map_err(|err| hf_async_api_error(Path::new(model_id), Some(file), revision, &err))
}

pub async fn read_model_file_range(
    model_id: &str,
    revision: &str,
    file: &str,
    range: Range<u64>,
    token_source: &crate::pipeline::TokenSource,
) -> Result<Vec<u8>> {
    let model_path = Path::new(model_id);
    if model_path.exists() {
        return read_local_file_range(&model_path.join(file), range).await;
    }

    if let Some(path) = offline_cache_repo(model_path, revision).get(file) {
        return read_local_file_range(&path, range).await;
    }

    if is_hf_hub_offline() {
        return Err(offline_missing_file_error(model_path, file, revision));
    }

    let url = model_file_url(model_id, revision, file);
    read_remote_file_range(&url, model_path, file, range, token_source).await
}

async fn read_local_file_range(path: &Path, range: Range<u64>) -> Result<Vec<u8>> {
    if range.start > range.end {
        return Err(anyhow!(
            "Invalid byte range {}..{} for `{}`.",
            range.start,
            range.end,
            path.display()
        ));
    }
    let len = usize::try_from(range.end - range.start)
        .map_err(|_| anyhow!("Byte range is too large for `{}`.", path.display()))?;
    let mut file = tokio::fs::File::open(path).await.map_err(|err| {
        anyhow!(
            "Could not open `{}` while reading byte range: {err}",
            path.display()
        )
    })?;
    file.seek(std::io::SeekFrom::Start(range.start)).await?;
    let mut data = vec![0u8; len];
    file.read_exact(&mut data).await?;
    Ok(data)
}

async fn read_remote_file_range(
    url: &str,
    model_id: &Path,
    file: &str,
    range: Range<u64>,
    token_source: &crate::pipeline::TokenSource,
) -> Result<Vec<u8>> {
    if range.start > range.end {
        return Err(anyhow!(
            "Invalid byte range {}..{} for `{file}`.",
            range.start,
            range.end
        ));
    }
    let len = usize::try_from(range.end - range.start)
        .map_err(|_| anyhow!("Byte range is too large for `{file}`."))?;
    if len == 0 {
        return Ok(Vec::new());
    }
    let http_range = format!("bytes={}-{}", range.start, range.end - 1);
    let mut request = range_http_client()
        .get(url)
        .header(reqwest::header::RANGE, http_range.clone());
    if let Some(token) = get_token(token_source)? {
        request = request.bearer_auth(token);
    }
    let response = request.send().await.map_err(|err| {
        anyhow!(
            "Failed to read byte range {http_range} from `{file}` for `{}`: {err}",
            model_id.display()
        )
    })?;

    if let Err(err) = response.error_for_status_ref() {
        return Err(anyhow!(
            "Failed to read byte range {http_range} from `{file}` for `{}`: {err}",
            model_id.display()
        ));
    }
    if response.status() != reqwest::StatusCode::PARTIAL_CONTENT
        && !response
            .headers()
            .contains_key(reqwest::header::CONTENT_RANGE)
    {
        return Err(anyhow!(
            "Server did not return a byte-range response for `{file}` in `{}`.",
            model_id.display()
        ));
    }

    let data = response.bytes().await?.to_vec();
    if data.len() != len {
        return Err(anyhow!(
            "Expected {len} bytes for range {http_range} from `{file}`, got {}.",
            data.len()
        ));
    }
    Ok(data)
}

fn range_http_client() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(reqwest::Client::new)
}

fn model_file_url(model_id: &str, revision: &str, file: &str) -> String {
    let endpoint = env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".to_string());
    let repo = Repo::with_revision(model_id.to_string(), RepoType::Model, revision.to_string());
    format!(
        "{}/{}/resolve/{}/{}",
        endpoint.trim_end_matches('/'),
        repo.url(),
        repo.url_revision(),
        file
    )
}

pub(crate) fn parse_status_code(message: &str) -> Option<u16> {
    let marker = "status code ";
    let (_, tail) = message.split_once(marker)?;
    let digits = tail
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>();
    digits.parse().ok()
}

pub(crate) fn api_error_status_code(err: &ApiError) -> Option<u16> {
    match err {
        ApiError::TooManyRetries(inner) => api_error_status_code(inner),
        _ => parse_status_code(&err.to_string()),
    }
}

pub(crate) fn async_api_error_status_code(err: &AsyncApiError) -> Option<u16> {
    match err {
        AsyncApiError::TooManyRetries(inner) => async_api_error_status_code(inner),
        _ => parse_status_code(&err.to_string()),
    }
}

pub(crate) fn should_propagate_api_error(err: &ApiError) -> bool {
    matches!(api_error_status_code(err), Some(401 | 403 | 404))
}

pub(crate) fn should_propagate_async_api_error(err: &AsyncApiError) -> bool {
    matches!(async_api_error_status_code(err), Some(401 | 403 | 404))
}

pub(crate) fn remote_issue_from_api_error(
    model_id: &Path,
    file: Option<&str>,
    revision: &str,
    err: &ApiError,
) -> RemoteAccessIssue {
    let target = match file {
        Some(file) => format!("`{file}` for `{}`", model_id.display()),
        None => format!("`{}`", model_id.display()),
    };
    RemoteAccessIssue {
        status_code: api_error_status_code(err),
        message: format!("Failed to access {target} (revision `{revision}`): {err}"),
        file: file.map(ToString::to_string),
        revision: revision.to_string(),
    }
}

pub(crate) fn hf_access_error(model_id: &Path, issue: &RemoteAccessIssue) -> anyhow::Error {
    hf_error(
        model_id,
        issue.file.as_deref(),
        &issue.revision,
        issue.status_code,
        Some(&issue.message),
    )
}

fn hf_error(
    model_id: &Path,
    file: Option<&str>,
    revision: &str,
    status_code: Option<u16>,
    detail: Option<&str>,
) -> anyhow::Error {
    let target = match file {
        Some(file) => format!(
            "`{file}` for `{}` (revision `{revision}`)",
            model_id.display()
        ),
        None => format!("`{}` (revision `{revision}`)", model_id.display()),
    };
    let check_hint = if file.is_some() {
        "Check the model ID, revision, access token, and requested filename."
    } else {
        "Check the model ID, revision, and access token."
    };
    let mut message = match status_code {
        Some(code @ (401 | 403)) => format!(
            "Could not access {target} on Hugging Face (HTTP {code}). Run `mistralrs login` or set `HF_TOKEN` if this is a private or gated repo."
        ),
        Some(404) => format!("Could not find {target} on Hugging Face (HTTP 404). {check_hint}"),
        Some(code) => format!("Failed to access {target} on Hugging Face (HTTP {code})."),
        None => format!("Failed to access {target} on Hugging Face."),
    };

    message.push(' ');
    message.push_str(&local_resolution_hint(model_id));

    if let Some(hint) = offline_cache_hint(model_id, revision, file) {
        message.push(' ');
        message.push_str(&hint);
    } else if status_code.is_none() {
        message.push_str(&format!(
            " If you are offline and have this repo cached, set `{HF_HUB_OFFLINE_ENV}=1`."
        ));
    }

    if let Some(detail) = detail {
        message.push_str(" Details: ");
        message.push_str(detail);
    }

    anyhow!(message)
}

pub(crate) fn hf_api_error(
    model_id: &Path,
    file: Option<&str>,
    revision: &str,
    err: &ApiError,
) -> anyhow::Error {
    let status_code = api_error_status_code(err);
    let detail = err.to_string();
    hf_error(model_id, file, revision, status_code, Some(&detail))
}

pub(crate) fn hf_async_api_error(
    model_id: &Path,
    file: Option<&str>,
    revision: &str,
    err: &AsyncApiError,
) -> anyhow::Error {
    let status_code = async_api_error_status_code(err);
    let detail = err.to_string();
    hf_error(model_id, file, revision, status_code, Some(&detail))
}

pub(crate) fn local_file_missing_error(model_id: &Path, file: &str) -> anyhow::Error {
    anyhow!(
        "File `{file}` was not found in local model directory `{}`. This model ID was treated as local because that directory exists, so Hugging Face was not queried for this file.",
        model_id.display()
    )
}

pub(crate) fn list_repo_files(
    api: &ApiRepo,
    model_id: &Path,
    should_error: bool,
    revision: &str,
) -> Result<Vec<String>> {
    if model_id.exists() {
        let listing = fs::read_dir(model_id).map_err(|err| {
            anyhow!(
                "Cannot list local model directory `{}`: {err}",
                model_id.display()
            )
        })?;
        let files = listing
            .filter_map(|entry| entry.ok())
            .filter_map(|entry| {
                entry
                    .path()
                    .file_name()
                    .and_then(|name| name.to_str())
                    .map(std::string::ToString::to_string)
            })
            .collect::<Vec<_>>();
        return Ok(files);
    }

    if is_hf_hub_offline() {
        let files = offline_snapshot_files(model_id, revision);
        if !files.is_empty() {
            return Ok(files);
        }
        if should_error {
            return Err(offline_missing_snapshot_error(model_id, revision));
        }
        warn!(
            "`{HF_HUB_OFFLINE_ENV}` is set and no local Hugging Face cache was found for `{}` (revision `{revision}`)",
            model_id.display()
        );
        return Ok(Vec::new());
    }

    match api.info() {
        Ok(repo) => {
            let files = repo
                .siblings
                .iter()
                .map(|x| x.rfilename.clone())
                .collect::<Vec<_>>();
            Ok(files)
        }
        Err(err) => {
            if should_error || should_propagate_api_error(&err) {
                Err(hf_api_error(model_id, None, revision, &err))
            } else {
                warn!(
                    "Could not get directory listing from Hugging Face for `{}`: {err}",
                    model_id.display()
                );
                Ok(Vec::new())
            }
        }
    }
}

pub(crate) async fn list_repo_files_async(
    api: &AsyncApiRepo,
    model_id: &Path,
    should_error: bool,
    revision: &str,
) -> Result<Vec<String>> {
    if model_id.exists() {
        let listing = fs::read_dir(model_id).map_err(|err| {
            anyhow!(
                "Cannot list local model directory `{}`: {err}",
                model_id.display()
            )
        })?;
        let files = listing
            .filter_map(|entry| entry.ok())
            .filter_map(|entry| {
                entry
                    .path()
                    .file_name()
                    .and_then(|name| name.to_str())
                    .map(std::string::ToString::to_string)
            })
            .collect::<Vec<_>>();
        return Ok(files);
    }

    if is_hf_hub_offline() {
        let files = offline_snapshot_files(model_id, revision);
        if !files.is_empty() {
            return Ok(files);
        }
        if should_error {
            return Err(offline_missing_snapshot_error(model_id, revision));
        }
        warn!(
            "`{HF_HUB_OFFLINE_ENV}` is set and no local Hugging Face cache was found for `{}` (revision `{revision}`)",
            model_id.display()
        );
        return Ok(Vec::new());
    }

    match api.info().await {
        Ok(repo) => {
            let files = repo
                .siblings
                .iter()
                .map(|x| x.rfilename.clone())
                .collect::<Vec<_>>();
            Ok(files)
        }
        Err(err) => {
            if should_error || should_propagate_async_api_error(&err) {
                Err(hf_async_api_error(model_id, None, revision, &err))
            } else {
                warn!(
                    "Could not get directory listing from Hugging Face for `{}`: {err}",
                    model_id.display()
                );
                Ok(Vec::new())
            }
        }
    }
}

/// How often [`cancellable_get`] polls [`download_cancel_flag`] while a blocking Hugging Face
/// file download runs on its own thread.
const DOWNLOAD_CANCEL_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Returns the process-wide flag that is flipped when this process receives SIGTERM, SIGINT, or
/// SIGQUIT. `spiced` (the runtime embedding this crate) installs its own signal handler for
/// graceful shutdown, but has no way to interrupt a blocking OS thread buried in this crate's
/// blocking Hugging Face download call. Registering our own handler here gives that download a
/// local, self-contained way to notice the same signals and return a clean cancellation error
/// instead of hanging indefinitely.
fn download_cancel_flag() -> &'static Arc<AtomicBool> {
    static FLAG: OnceLock<Arc<AtomicBool>> = OnceLock::new();
    FLAG.get_or_init(|| {
        let flag = Arc::new(AtomicBool::new(false));
        for sig in TERM_SIGNALS {
            if let Err(err) = signal_hook::flag::register(*sig, Arc::clone(&flag)) {
                warn!(
                    "Could not register a termination-signal handler for cancelling in-flight Hugging Face downloads: {err}"
                );
            }
        }
        flag
    })
}

/// Outcome of [`cancellable_get`] when the blocking request did not complete because it was
/// cancelled by a termination signal, as opposed to failing on its own.
enum CancellableGetError {
    Api(ApiError),
    Cancelled,
}

/// Waits for `handle` to finish, polling `should_terminate` every
/// [`DOWNLOAD_CANCEL_POLL_INTERVAL`]. Returns the handle's result if it finishes first, or `None`
/// if a termination signal fires first -- in which case `handle` is dropped and abandoned to
/// finish (or fail) on its own; its eventual result is discarded.
fn poll_until_finished_or_cancelled<T: Send + 'static>(
    handle: thread::JoinHandle<T>,
    should_terminate: &AtomicBool,
) -> Option<T> {
    loop {
        if handle.is_finished() {
            return Some(match handle.join() {
                Ok(result) => result,
                Err(panic) => std::panic::resume_unwind(panic),
            });
        }
        if should_terminate.load(AtomicOrdering::SeqCst) {
            return None;
        }
        thread::sleep(DOWNLOAD_CANCEL_POLL_INTERVAL);
    }
}

/// Runs the blocking `ApiRepo::get(file)` call on a dedicated thread so this function can poll
/// [`download_cancel_flag`] and return promptly on SIGTERM/SIGINT/SIGQUIT instead of blocking
/// until a (potentially very large) download finishes. The spawned thread owns an `Arc` clone of
/// `api`, so if cancellation fires first, the download thread is simply abandoned to finish (or
/// fail) on its own; its result is discarded.
fn cancellable_get(
    api: &Arc<ApiRepo>,
    file: &str,
) -> std::result::Result<PathBuf, CancellableGetError> {
    let handle = {
        let api = Arc::clone(api);
        let file = file.to_string();
        thread::spawn(move || api.get(&file))
    };

    match poll_until_finished_or_cancelled(handle, download_cancel_flag()) {
        Some(result) => result.map_err(CancellableGetError::Api),
        None => Err(CancellableGetError::Cancelled),
    }
}

fn hf_download_cancelled_error(model_id: &Path, file: &str, revision: &str) -> anyhow::Error {
    anyhow!(
        "Download of `{file}` for `{}` (revision `{revision}`) was cancelled because mistral.rs received a termination signal (SIGTERM/SIGINT/SIGQUIT).",
        model_id.display()
    )
}

pub(crate) fn get_file(
    api: &Arc<ApiRepo>,
    model_id: &Path,
    file: &str,
    revision: &str,
) -> Result<PathBuf> {
    if model_id.exists() {
        let path = model_id.join(file);
        if !path.exists() {
            return Err(local_file_missing_error(model_id, file));
        }
        trace!("Loading `{file}` locally at `{}`", path.display());
        return Ok(path);
    }

    if is_hf_hub_offline() {
        if let Some(path) = offline_cache_repo(model_id, revision).get(file) {
            trace!(
                "Loading `{file}` from local Hugging Face cache at `{}` (offline mode)",
                path.display()
            );
            return Ok(path);
        }
        return Err(offline_missing_file_error(model_id, file, revision));
    }

    match cancellable_get(api, file) {
        Ok(path) => Ok(path),
        Err(CancellableGetError::Api(err)) => {
            Err(hf_api_error(model_id, Some(file), revision, &err))
        }
        Err(CancellableGetError::Cancelled) => {
            Err(hf_download_cancelled_error(model_id, file, revision))
        }
    }
}

/// Like [`get_file`] but returns `Ok(None)` (instead of an error) when the file is genuinely missing, and used with `HF_HUB_OFFLINE`.
pub(crate) fn try_get_file(
    api: &Arc<ApiRepo>,
    model_id: &Path,
    file: &str,
    revision: &str,
) -> std::result::Result<Option<PathBuf>, ApiError> {
    if model_id.exists() {
        let path = model_id.join(file);
        if path.exists() {
            trace!("Loading `{file}` locally at `{}`", path.display());
            return Ok(Some(path));
        }
        return Ok(None);
    }

    if is_hf_hub_offline() {
        return Ok(offline_cache_repo(model_id, revision).get(file));
    }

    match cancellable_get(api, file) {
        Ok(p) => Ok(Some(p)),
        Err(CancellableGetError::Api(err)) => match api_error_status_code(&err) {
            Some(404) => Ok(None),
            _ => Err(err),
        },
        Err(CancellableGetError::Cancelled) => Err(ApiError::IoError(std::io::Error::new(
            std::io::ErrorKind::Interrupted,
            format!(
                "Download of `{file}` was cancelled because mistral.rs received a termination signal (SIGTERM/SIGINT/SIGQUIT)."
            ),
        ))),
    }
}

pub(crate) async fn try_get_file_async(
    api: &AsyncApiRepo,
    model_id: &Path,
    file: &str,
    revision: &str,
) -> std::result::Result<Option<PathBuf>, AsyncApiError> {
    if model_id.exists() {
        let path = model_id.join(file);
        if path.exists() {
            trace!("Loading `{file}` locally at `{}`", path.display());
            return Ok(Some(path));
        }
        return Ok(None);
    }

    if is_hf_hub_offline() {
        return Ok(offline_cache_repo(model_id, revision).get(file));
    }

    match api.get(file).await {
        Ok(p) => Ok(Some(p)),
        Err(err) => match async_api_error_status_code(&err) {
            Some(404) => Ok(None),
            _ => Err(err),
        },
    }
}

/// Best-effort file listing for a HF repo. Returns `None` on 404, API failure,
/// or offline-without-cache. Quiet by design: callers choose what to log.
pub fn probe_hf_repo_files(
    model_id: &str,
    revision: &str,
    token_source: &crate::pipeline::TokenSource,
) -> Option<Vec<String>> {
    if is_hf_hub_offline() {
        let files = offline_snapshot_files(Path::new(model_id), revision);
        return (!files.is_empty()).then_some(files);
    }

    let repo = build_api(token_source, false)
        .ok()?
        .repo(Repo::with_revision(
            model_id.to_string(),
            RepoType::Model,
            revision.to_string(),
        ));
    repo.info()
        .ok()
        .map(|info| info.siblings.into_iter().map(|s| s.rfilename).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A termination signal that is already set before the "download" thread has any chance to
    /// finish must win the race: [`poll_until_finished_or_cancelled`] returns `None` (the
    /// cancelled outcome) instead of waiting for the thread. Uses a thread that sleeps far longer
    /// than the poll interval so the test is deterministic without touching the network or the
    /// process-wide signal-registered flag from [`download_cancel_flag`].
    #[test]
    fn poll_until_finished_or_cancelled_returns_none_when_already_cancelled() {
        let should_terminate = AtomicBool::new(true);
        let handle = thread::spawn(|| {
            thread::sleep(Duration::from_secs(5));
            "unused"
        });

        let result = poll_until_finished_or_cancelled(handle, &should_terminate);

        assert!(
            result.is_none(),
            "expected cancellation to win the race, got {result:?}"
        );
    }

    /// When the handle finishes before any termination signal arrives, its result is returned
    /// normally.
    #[test]
    fn poll_until_finished_or_cancelled_returns_result_when_not_cancelled() {
        let should_terminate = AtomicBool::new(false);
        let handle = thread::spawn(|| 42);

        let result = poll_until_finished_or_cancelled(handle, &should_terminate);

        assert_eq!(result, Some(42));
    }
}
