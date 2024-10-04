use anyhow::Result;
use mistralrs::{
    ChatCompletionResponse, IsqType, PagedAttentionMetaBuilder, TextMessageRole, TextMessages,
    TextModelBuilder, Usage,
};

const N_REQUESTS: usize = 10;

#[tokio::main]
async fn main() -> Result<()> {
    let model = TextModelBuilder::new("microsoft/Phi-3.5-mini-instruct")
        .with_isq(IsqType::Q8_0)
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

    let mut handles = Vec::new();
    for _ in 0..N_REQUESTS {
        handles.push(model.send_chat_request(messages.clone()));
    }
    let responses = futures::future::join_all(handles)
        .await
        .into_iter()
        .collect::<Result<Vec<_>>>()?;

    let mut max_prompt = f32::MIN;
    let mut max_completion = f32::MIN;

    for response in responses {
        let ChatCompletionResponse {
            usage:
                Usage {
                    avg_compl_tok_per_sec,
                    avg_prompt_tok_per_sec,
                    ..
                },
            ..
        } = response;
        dbg!(avg_compl_tok_per_sec, avg_prompt_tok_per_sec);
        if avg_compl_tok_per_sec > max_prompt {
            max_prompt = avg_prompt_tok_per_sec;
        }
        if avg_compl_tok_per_sec > max_completion {
            max_completion = avg_compl_tok_per_sec;
        }
    }
    println!("Individual sequence stats: {max_prompt} max PP T/s, {max_completion} max TG T/s");

    Ok(())
}
