import streamlit as st
import boto3
import json


# ---- Session state init ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---- Bedrock call ----
client_bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")  # uses your AWS profile/ENV

def call_bedrock(prompt: str) -> str:
    """
    Sends a user prompt to the Anthropic Claude 3.5 Haiku model via Bedrock Converse API
    and returns the assistant's text response.
    """
    # Use the inference profile ARN for Haiku (profile-only invocation)
    model_id = "arn:aws:bedrock:us-east-1:362934839387:inference-profile/us.anthropic.claude-3-5-haiku-20241022-v1:0"

    # Build the message structure
    messages = [
        {
            "role": "user",
            "content": [{"text": prompt}]
        }
    ]

    # Define inference configuration parameters
    inference_config = {
        "maxTokens": 100,
        "temperature": 0.9,
        "topP": 0.75
        # Additional parameters like 'topK' can also be added here
    }
    try:
        # Call Bedrock using the Converse API
        response = client_bedrock.converse(
            modelId=model_id,
            messages=messages,
            inferenceConfig=inference_config
        )

        # Extract assistant’s response text
        return response["output"]["message"]["content"][0]["text"]

    except Exception as e:
        print(f"ERROR invoking model: {e}")
        raise


# Start a conversation with the user message.
user_message = "Describe the purpose of a 'hello world' program in one line."
conversation = [
    {
        "role": "user",
        "content": [{"text": user_message}],
    }
]

st.title("Bedrock Chatbot")

# ---- UI ----
with st.form(key="chat_form"):
    user_input = st.text_input("You:")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    st.session_state.chat_history.append(("User", user_input))
    try:
        bot_response = call_bedrock(user_input)
    except Exception as e:
        bot_response = f"Error calling Bedrock model: {e}"
    st.session_state.chat_history.append(("Bot", bot_response))

st.markdown("### Conversation History")
for speaker, message in st.session_state.chat_history:
    st.markdown(f"**{speaker}:** {message}")
