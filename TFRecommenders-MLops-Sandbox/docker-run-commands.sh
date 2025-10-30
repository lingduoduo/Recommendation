docker build -t genai-chatbot .

docker run -p 8501:8501 -v ~/.aws:/root/.aws genai-chatbot

# REGION=us-east-1
# ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# aws ecr get-login-password --region $REGION \
# | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

# aws ecr create-repository --repository-name genai-apps --region $REGION || true

# docker build -t genai-chatbot:latest .

# docker tag genai-chatbot:latest ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/genai-apps:genai-chatbot

# docker push ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/genai-apps:genai-chatbot

aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 362934839387.dkr.ecr.us-east-1.amazonaws.com

docker build -t genai-chatbot .

docker tag genai-chatbot:latest 362934839387.dkr.ecr.us-east-1.amazonaws.com/genai-chatbot:latest

docker push 362934839387.dkr.ecr.us-east-1.amazonaws.com/genai-chatbot:latest