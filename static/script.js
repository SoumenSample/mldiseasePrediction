function uploadImage() {
    let fileInput = document.getElementById("fileInput");
    if (fileInput.files.length === 0) {
        alert("Please select an image.");
        return;
    }

    let formData = new FormData();
    formData.append("file", fileInput.files[0]);

    fetch("/predict", { method: "POST", body: formData })
        .then(response => response.json())
        .then(data => {
            let resultDiv = document.getElementById("result");
            if (data.error) {
                resultDiv.innerHTML = `<p class="error">${data.error}</p>`;
            } else {
                resultDiv.innerHTML = `<p><strong>Leaf Name:</strong> ${data.leaf_name}</p>
                                       <p><strong>Status:</strong> ${data.status}</p>
                                       <p><strong>Confidence:</strong> ${data.confidence}</p>`;
                let img = document.getElementById("uploadedImage");
                img.src = data.image_url;
                img.style.display = "block";
            }
        })
        .catch(error => console.error("Error:", error));
}
