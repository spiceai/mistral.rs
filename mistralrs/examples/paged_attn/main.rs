use anyhow::Result;
use mistralrs::{
    IsqType, MemoryGpuConfig, PagedAttentionMetaBuilder, TextMessageRole, TextMessages,
    TextModelBuilder,
};

#[tokio::main]
async fn main() -> Result<()> {
    let model = TextModelBuilder::new("microsoft/Phi-3.5-mini-instruct")
        .with_isq(IsqType::Q8_0)
        .with_logging()
        .with_paged_attn(|| {
            PagedAttentionMetaBuilder::default()
                .with_block_size(32)
                .with_gpu_memory(MemoryGpuConfig::ContextSize(1024))
                .build()
        })?
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
