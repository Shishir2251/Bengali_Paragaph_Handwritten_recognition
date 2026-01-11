const uploadBox = document.getElementById('uploadBox');
const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const uploadContent = document.getElementById('uploadContent');
const recognizeBtn = document.getElementById('recognizeBtn');
const result = document.getElementById('result');
const resultText = document.getElementById('resultText');
const copyBtn = document.getElementById('copyBtn');

let selectedFile = null;

uploadBox.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            preview.src = e.target.result;
            preview.style.display = 'block';
            uploadContent.style.display = 'none';
            recognizeBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }
});

recognizeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    
    recognizeBtn.textContent = 'Processing...';
    recognizeBtn.disabled = true;
    
    const formData = new FormData();
    formData.append('image', selectedFile);
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            resultText.textContent = data.text;
            result.style.display = 'block';
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        recognizeBtn.textContent = 'Recognize Text';
        recognizeBtn.disabled = false;
    }
});

copyBtn.addEventListener('click', () => {
    navigator.clipboard.writeText(resultText.textContent);
    copyBtn.textContent = 'Copied!';
    setTimeout(() => {
        copyBtn.textContent = 'Copy';
    }, 2000);
});