#!/bin/bash

# --- 0. AUTOMAATNE VÕTME TUVASTAMINE ---
# Leiab esimese Key Pairi nime sinu AWS kontolt
KEY_NAME=$(aws ec2 describe-key-pairs --query 'KeyPairs[0].KeyName' --output text)

if [ "$KEY_NAME" == "None" ] || [ -z "$KEY_NAME" ]; then
    echo "VIGA: Sul ei ole ühtegi Key Pairi! Loo see AWS konsoolis (EC2 -> Key Pairs)."
    exit 1
fi

# --- KONFIGURATSIOON ---
INSTANCE_NAME="AI-Kursatoo-Server"
INSTANCE_TYPE="i4i.large"
IMAGE_ID="ami-077c6fac5ef663f46"  # Ubuntu 24.04 Deep Learning AMI
VOLUME_SIZE=40                    # Piisav varu mudelitele ja logidele
REPO_URL="https://github.com/alarj/Double_Check_AI.git"

echo "--- 1. Kasutan võtit: $KEY_NAME ---"

echo "--- 2. Turvagrupi loomine ja portide avamine ---"
SG_NAME="AI-Security-Group-$(date +%s)"
SG_ID=$(aws ec2 create-security-group \
    --group-name "$SG_NAME" \
    --description "Port 8501 Streamlit ja 22 SSH" \
    --query 'GroupId' --output text)

aws ec2 authorize-security-group-ingress --group-id $SG_ID --protocol tcp --port 22 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-id $SG_ID --protocol tcp --port 8501 --cidr 0.0.0.0/0

echo "--- 3. Käivitan masina (AMI: $IMAGE_ID, Ketas: ${VOLUME_SIZE}GB) ---"
INSTANCE_INFO=$(aws ec2 run-instances \
    --image-id $IMAGE_ID \
    --count 1 \
    --instance-type $INSTANCE_TYPE \
    --key-name "$KEY_NAME" \
    --security-group-ids $SG_ID \
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$VOLUME_SIZE,\"VolumeType\":\"gp3\"}}]" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$INSTANCE_NAME}]" \
    --user-data "#!/bin/bash
        # Logime kogu protsessi faili /var/log/user-data.log
        exec > /var/log/user-data.log 2>&1
        
        echo \"--- Algab süsteemi seadistamine ---\"

        # 1. Veendume, et eelinstalleeritud Docker töötab
        systemctl start docker
        systemctl enable docker
        usermod -aG docker ubuntu
        
        # 2. Koodi allalaadimine GitHubist
        cd /home/ubuntu
        rm -rf Double_Check_AI
        git clone $REPO_URL
        
        # 3. Liigume 'logic' kataloogi, kus asub docker-compose.yml
        cd Double_Check_AI/logic
        
        # 4. Käivitame rakenduse (kasutame süsteemset 'docker compose' käsku)
        docker compose up -d
        
        # 5. Tõmbame AI mudelid (Ollama konteinerisse)
        echo \"Ootan 30 sekundit, et Ollama startiks...\"
        sleep 30
        docker exec ollama ollama pull phi3:mini
        docker exec ollama ollama pull llama3:8b
        docker exec ollama ollama pull gemma2:2b
        
        echo \"--- Paigaldus lõpetatud! ---\"
        " \
    --query 'Instances[0].[InstanceId, PublicIpAddress]' --output text)

ID=$(echo $INSTANCE_INFO | awk '{print $1}')
IP=$(echo $INSTANCE_INFO | awk '{print $2}')

echo "------------------------------------------------"
echo "✅ INFRASTRUKTUUR ON LOODUD!"
echo "ID: $ID"
echo "IP: $IP"
echo "Ketas: ${VOLUME_SIZE}GB"
echo "------------------------------------------------"
echo "Süsteem seadistab end taustal (ca 10 min)."
echo "Valmis rakendus ilmub aadressile: http://$IP:8501"
echo "------------------------------------------------"
echo "Logide jälgimiseks SSH-s: tail -f /var/log/user-data.log"
