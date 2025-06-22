import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# 設定頁面配置
st.set_page_config(
    page_title="手寫數字生成器",
    page_icon="✍️",
    layout="wide"
)

# VAE 模型架構（與訓練時相同）
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=32, num_classes=10):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # 編碼器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # 潛在空間參數
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # 解碼器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x, labels):
        labels_onehot = F.one_hot(labels, self.num_classes).float()
        x_with_labels = torch.cat([x, labels_onehot], dim=1)
        h = self.encoder(x_with_labels)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, labels):
        labels_onehot = F.one_hot(labels, self.num_classes).float()
        z_with_labels = torch.cat([z, labels_onehot], dim=1)
        return self.decoder(z_with_labels)
    
    def forward(self, x, labels):
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, labels)
        return recon_x, mu, logvar
    
    def generate(self, labels, num_samples=1):
        """生成指定標籤的圖片"""
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim)
            generated = self.decode(z, labels)
            return generated.view(-1, 28, 28)

# 載入模型
@st.cache_resource
def load_model():
    model = VAE()
    try:
        # 嘗試載入訓練好的模型
        model.load_state_dict(torch.load('mnist_vae_model.pth', map_location='cpu'))
        model.eval()
        return model, True
    except FileNotFoundError:
        # 如果沒有訓練好的模型，使用隨機權重
        st.warning("未找到訓練好的模型檔案，使用隨機權重進行展示")
        model.eval()
        return model, False

# 生成圖片
def generate_digit_images(model, digit, num_samples=5):
    labels = torch.full((num_samples,), digit, dtype=torch.long)
    generated_images = model.generate(labels, num_samples=num_samples)
    return generated_images.numpy()

# 將 numpy 陣列轉換為 PIL 圖片
def numpy_to_pil(img_array):
    # 正規化到 0-255 範圍
    img_array = (img_array * 255).astype(np.uint8)
    return Image.fromarray(img_array, mode='L')

# 主應用程式
def main():
    st.title("🔢 手寫數字生成器")
    st.subheader("使用變分自編碼器 (VAE) 生成 MNIST 風格的手寫數字")
    
    # 載入模型
    model, model_loaded = load_model()
    
    if model_loaded:
        st.success("✅ 模型載入成功！")
    else:
        st.info("ℹ️ 使用隨機權重模型進行展示")
    
    # 側邊欄控制
    st.sidebar.header("控制面板")
    
    # 選擇要生成的數字
    selected_digit = st.sidebar.selectbox(
        "選擇要生成的數字 (0-9):",
        options=list(range(10)),
        index=2
    )
    
    # 生成按鈕
    if st.sidebar.button("🎨 生成圖片", type="primary"):
        with st.spinner("正在生成圖片..."):
            # 生成圖片
            generated_images = generate_digit_images(model, selected_digit, num_samples=5)
            
            # 顯示結果
            st.header(f"生成的數字 {selected_digit} 圖片")
            
            # 使用列布局顯示圖片
            cols = st.columns(5)
            
            for i, img in enumerate(generated_images):
                with cols[i]:
                    # 轉換為 PIL 圖片並顯示
                    pil_img = numpy_to_pil(img)
                    st.image(pil_img, caption=f"樣本 {i+1}", width=150)
                    
                    # 提供下載按鈕
                    buf = io.BytesIO()
                    pil_img.save(buf, format='PNG')
                    buf.seek(0)
                    
                    st.download_button(
                        label="下載",
                        data=buf,
                        file_name=f"digit_{selected_digit}_sample_{i+1}.png",
                        mime="image/png",
                        key=f"download_{selected_digit}_{i}"
                    )
    
    # 資訊區塊
    with st.expander("📋 關於這個應用程式"):
        st.markdown("""
        ### 功能特色
        - 🎯 **選擇數字**: 可以選擇要生成的數字 (0-9)
        - 🖼️ **生成圖片**: 一次生成 5 張不同的手寫數字圖片
        - 💾 **下載功能**: 可以下載生成的圖片
        - 🤖 **AI 模型**: 使用變分自編碼器 (VAE) 架構
        
        ### 技術細節
        - **模型**: 變分自編碼器 (Variational Autoencoder, VAE)
        - **資料集**: MNIST 手寫數字資料集
        - **框架**: PyTorch
        - **圖片尺寸**: 28x28 像素，灰階
        
        ### 使用方法
        1. 在左側控制面板選擇要生成的數字
        2. 點擊「生成圖片」按鈕
        3. 查看生成的 5 張圖片
        4. 可以下載喜歡的圖片
        """)
    
    # 模型架構資訊
    if st.checkbox("顯示模型架構"):
        st.subheader("🏗️ 模型架構")
        st.code("""
        VAE 架構:
        - 編碼器: Linear(794) -> ReLU -> Linear(256) -> ReLU
        - 潛在空間: 32 維
        - 解碼器: Linear(42) -> ReLU -> Linear(256) -> ReLU -> Linear(784) -> Sigmoid
        - 條件輸入: 使用 one-hot 編碼的標籤
        """)

if __name__ == "__main__":
    main()
