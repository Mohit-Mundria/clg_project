#!/bin/bash
# KisanAI AWS EC2 Deployment Setup Script
# Run this on your Ubuntu EC2 instance

echo "================================================="
echo " KisanAI AWS Deployment Setup Script"
echo "================================================="

# 1. Update system packages
echo "[+] Updating system packages..."
sudo apt-get update -y
sudo apt-get upgrade -y

# 2. Install Docker
if ! command -v docker &> /dev/null; then
    echo "[+] Installing Docker..."
    sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" -y
    sudo apt-get update -y
    sudo apt-get install -y docker-ce
    sudo usermod -aG docker ubuntu
    echo "[OK] Docker installed successfully."
else
    echo "[OK] Docker is already installed."
fi

# 3. Install Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "[+] Installing Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.24.5/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    echo "[OK] Docker Compose installed successfully."
else
    echo "[OK] Docker Compose is already installed."
fi

echo "================================================="
echo "[SUCCESS] Environment is ready!"
echo "NOTE: Please log out of the EC2 terminal and log back in for Docker group changes to take effect."
echo "After logging back in, navigate to the project directory and run:"
echo "   docker-compose up -d --build"
echo "================================================="
