const leafInput = document.getElementById("leafInput");
const previewImg = document.getElementById("previewImg");

leafInput.addEventListener("change", function () {
  const file = this.files[0];
  if (file) {
    previewImg.src = URL.createObjectURL(file);
    previewImg.style.display = "block";
  }
});