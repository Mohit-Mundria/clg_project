# KisanAI - AWS EC2 Deployment Guide

This guide will walk you through deploying your KisanAI project on an AWS EC2 instance using Docker. This is the most professional and robust way to deploy web applications.

> [!IMPORTANT]
> Make sure you have your AWS account ready and your `.env` file containing the GROQ_API_KEY and WEATHER_API_KEY locally.

## Step 1: Launch an EC2 Instance on AWS

1. Log in to the [AWS Management Console](https://console.aws.amazon.com/).
2. Search for **EC2** and go to the EC2 Dashboard.
3. Click on **Launch Instance**.
4. **Name and tags:** Enter a name like `KisanAI-Server`.
5. **Application and OS Images (Amazon Machine Image):** Select **Ubuntu**, and choose `Ubuntu Server 24.04 LTS (HVM)` or `22.04 LTS`.
6. **Instance type:** Since we are running ML training and models, select at least **t2.small** or **t2.medium**. (Note: `t2.micro` might freeze during ML training because of memory limits, but you can try it if you are strictly on the Free Tier).
7. **Key pair (login):** Click **Create new key pair**. 
   - Name it `kisanai-key`.
   - Key pair type: **RSA**
   - Private key file format: **.pem**
   - Click Create (this will download the `.pem` file to your computer. Keep it safe!).
8. **Network settings:**
   - Check **Allow SSH traffic from** (Anywhere 0.0.0.0/0)
   - Check **Allow HTTP traffic from the internet** (This allows people to access your app on port 80)
   - Check **Allow HTTPS traffic from the internet**
9. **Configure storage:** 16 GB to 20 GB gp3 is recommended.
10. Click **Launch Instance**.

## Step 2: Connect to your EC2 Instance

1. Once the instance is running, select it and copy its **Public IPv4 address**.
2. Open your terminal (or Command Prompt/PowerShell) on your local computer where your `.pem` key is downloaded.
3. Change permissions of your key (Mac/Linux only):
   ```bash
   chmod 400 kisanai-key.pem
   ```
4. Connect via SSH:
   ```bash
   ssh -i "kisanai-key.pem" ubuntu@<YOUR_EC2_PUBLIC_IP>
   ```

## Step 3: Upload the Project to EC2

We need to upload your project files from your local computer to the EC2 server. Keep your SSH terminal open, and open a **new terminal window** on your local machine.

Run this command from the parent folder of `clg_project`:
```bash
scp -i "path/to/kisanai-key.pem" -r clg_project ubuntu@<YOUR_EC2_PUBLIC_IP>:~/clg_project
```
*(Make sure to replace the path to your `.pem` file and the `<YOUR_EC2_PUBLIC_IP>`)*

## Step 4: Run the Deployment Setup

Go back to your SSH terminal (where you are logged into Ubuntu).

1. Navigate to the project directory:
   ```bash
   cd ~/clg_project
   ```
2. Make the deployment script executable and run it:
   ```bash
   chmod +x deploy.sh
   ./deploy.sh
   ```
   *This script installs Docker and Docker Compose automatically.*

3. **Log out and log back in** to apply Docker permissions:
   ```bash
   exit
   ```
   Then reconnect: `ssh -i "kisanai-key.pem" ubuntu@<YOUR_EC2_PUBLIC_IP>`

## Step 5: Start the Server

1. Navigate to the project again:
   ```bash
   cd ~/clg_project
   ```
2. Make sure your `.env` file is uploaded. If it isn't, you can create it:
   ```bash
   nano .env
   ```
   Paste your variables (`GROQ_API_KEY`, etc.), press `CTRL + O`, `Enter`, and `CTRL + X` to save and exit.
3. Start the application using Docker Compose:
   ```bash
   docker-compose up -d --build
   ```

> [!TIP]
> The `--build` flag builds the Docker image and runs the `train_models.py` script automatically inside the container to prepare the ML models.
> The `-d` flag runs the server in the background so it stays online even when you close the terminal.

## Step 6: Access your Application

Your app is now live! Open any web browser and type your EC2 Public IP address:
```
http://<YOUR_EC2_PUBLIC_IP>
```

### Useful Commands to Manage the Server
- **View logs:** `docker-compose logs -f`
- **Stop server:** `docker-compose down`
- **Restart server:** `docker-compose restart`
