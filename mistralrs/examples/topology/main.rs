use anyhow::Result;
use mistralrs::{
    IsqType, LayerTopology, PagedAttentionMetaBuilder, TextMessageRole, TextMessages,
    TextModelBuilder, Topology,
};

#[tokio::main]
async fn main() -> Result<()> {
    let model = TextModelBuilder::new("microsoft/Phi-3.5-mini-instruct")
        .with_isq(IsqType::Q8_0)
        .with_topology(
            Topology::empty()
                .with_range(
                    0..8,
                    LayerTopology {
                        isq: Some(IsqType::Q3K),
                        device: None,
                    },
                )
                .with_range(
                    8..16,
                    LayerTopology {
                        isq: Some(IsqType::Q4K),
                        device: None,
                    },
                )
                .with_range(
                    16..24,
                    LayerTopology {
                        isq: Some(IsqType::Q6K),
                        device: None,
                    },
                )
                .with_range(
                    24..32,
                    LayerTopology {
                        isq: Some(IsqType::Q8_0),
                        device: None,
                    },
                ),
        )
        .with_logging()
        .with_paged_attn(|| PagedAttentionMetaBuilder::default().build())?
        .build()
        .await?;

    let messages = TextMessages::new()
        .add_message(
            TextMessageRole::System,
            "You are an AI agent with a specialty in programming.",
        )
        .add_message(
            TextMessageRole::User,
            "Hello! How are you? Please write generic binary search function in Rust.",
        );

    let response = model.send_chat_request(messages).await?;

    println!("{}", response.choices[0].message.content.as_ref().unwrap());
    dbg!(
        response.usage.avg_prompt_tok_per_sec,
        response.usage.avg_compl_tok_per_sec
    );

    Ok(())
}
