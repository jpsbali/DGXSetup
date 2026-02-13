# Setup ngrok 

Download ngrok
https://ngrok.com/download/linux

Setup an account and enable MFA and follow directions to setup your endpoint at 
https://dashboard.ngrok.com/get-started/setup/linux

Sample Commands :
1. Run following command on source system:
  curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
  | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null \
  && echo "deb https://ngrok-agent.s3.amazonaws.com bookworm main" \
  | sudo tee /etc/apt/sources.list.d/ngrok.list \
  && sudo apt update \
  && sudo apt install ngrok

2. ngrok config add-authtoken .....
3. ngrok http <port> 
4.  You can now test ngrok endpoint by using the forwarding information displayed on ngrok terminal (sample below):
Web Interface                 http://127.0.0.1:4040 
Forwarding                    https://<XYZ>.ngrok-free.app -> http://localhost:<Port>    
5. Secure your endpoint using OAuth
