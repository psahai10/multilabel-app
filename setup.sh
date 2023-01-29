mkdir -p ~/.streamlit/
echo "[general]
email = \"email@com\"
" > ~/.streamlit/credentials.toml
echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml
# echo "[password]
# psahai = \"streamlit123\"
# kambert = \"streamlit123\"
# " > ~/.streamlit/secrets.toml