#!/bin/bash

# --- 0. AUTOMAATNE VÕTME TUVASTAMINE ---
# Haarab esimese leitud KeyPair nime kursakaaslase kontolt
KEY_NAME=$(aws ec2 describe-key-pairs --query 'KeyPairs[0].KeyName' --output text)

if [ "$KEY_NAME" == "None" ] || [ -z "$KEY_NAME" ]; then
    echo "VIGA: Sul ei ole ühtegi Key Pairi! Loo see AWS konsoolis (EC2 -> Key Pairs)."
    exit 1
fi

# --- KONFIGURATSIOON ---
INSTANCE_NAME="AI-Kursatoo-Server"
INSTANCE_TYPE="i4i.large"
IMAGE_ID="ami-077c6fac5ef663f46"  # Ubuntu 24.04 Deep Learning AMI
VOLUME_SIZE=32
REPO_URL="https://github.com/alarj/Double_Check_AI.git"

echo "--- 1. Kasutan võtit: $KEY_NAME ---"
echo "--- 2. Loon turvagrupi ja avan pordid ---"
SG_NAME="AI-Security-Group-$(date +%s)"
SG_ID=$(aws ec2 create-security-group \
    --group-name "$SG_NAME" \
    --description "Port 8501 Streamlit ja 22 SSH" \
    --query 'GroupId' --output text)

aws ec2 authorize-security-group-ingress --group-id $SG_ID --protocol tcp --port 22 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-id $SG_ID --protocol tcp --port 8501 --cidr 0.0.0.0/0

echo "--- 3. Käivitan masina (AMI: $IMAGE_ID) ---"
INSTANCE_INFO=$(aws ec2 run-instances \
    --image-id $IMAGE_ID \
    --count 1 \
    --instance-type $INSTANCE_TYPE \
    --key-name "$KEY_NAME" \
    --security-group-ids $SG_ID \
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$VOLUME_SIZE}}]" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$INSTANCE_NAME}]" \
    --user-data "#!/bin/bash
        exec > /var/log/user-data.log 2>&1
        
        # Süsteemi ettevalmistus
        apt-get update -y
        apt-get install -y docker.io docker-compose git
        systemctl start docker
        usermod -aG docker ubuntu
        
        # Koodi allalaadimine MAIN harust
        cd /home/ubuntu
        git clone $REPO_URL
        cd Double_Check_AI
        
        # Käivitamine ja mudelite laadimine
        docker-compose up -d
        sleep 30
        docker exec ollama ollama pull phi3:mini
        docker exec ollama ollama pull llama3:8b
        " \
    --query 'Instances[0].[InstanceId, PublicIpAddress]' --output text)

ID=$(echo $INSTANCE_INFO | awk '{print $1}')
IP=$(echo $INSTANCE_INFO | awk '{print $2}')

echo "------------------------------------------------"
echo "✅ INFRASTRUKTUUR ON LOODUD!"
echo "ID: $ID"
echo "IP: $IP"
echo "Kood: Võetud GitHubi MAIN harust"
echo "------------------------------------------------"
echo "Süsteem seadistab end taustal (ca 10 min)."
echo "Valmis rakendus: http://$IP:8501"