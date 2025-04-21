document.addEventListener("DOMContentLoaded", () => {
  // File input handling
  const imageInput = document.getElementById("imageInput")
  const fileName = document.getElementById("fileName")

  if (imageInput) {
    imageInput.addEventListener("change", function () {
      if (this.files && this.files[0]) {
        fileName.textContent = this.files[0].name
      } else {
        fileName.textContent = "No file selected"
      }
    })
  }

  // Modal handling
  const modal = document.getElementById("addDocumentModal")
  const btn = document.getElementById("addDocumentBtn")
  const span = document.getElementsByClassName("close")[0]

  if (btn && modal && span) {
    btn.onclick = () => {
      modal.style.display = "block"
    }

    span.onclick = () => {
      modal.style.display = "none"
    }

    window.onclick = (event) => {
      if (event.target == modal) {
        modal.style.display = "none"
      }
    }
  }

  // Document input handling
  const documentInput = document.getElementById("documentInput")

  if (documentInput) {
    documentInput.addEventListener("change", function () {
      const fileSize = this.files[0].size
      const maxSize = 50 * 1024 * 1024 // 50MB

      if (fileSize > maxSize) {
        alert("File size exceeds 50MB limit")
        this.value = ""
      }

      const fileType = this.files[0].type
      if (fileType !== "application/pdf") {
        alert("Only PDF files are allowed")
        this.value = ""
      }
    })
  }

  // Notification handling
  const closeButtons = document.querySelectorAll(".close-notification")

  closeButtons.forEach((button) => {
    button.addEventListener("click", function () {
      const notification = this.parentElement
      notification.style.opacity = "0"
      setTimeout(() => {
        notification.remove()
      }, 500)
    })
  })

  // Auto-hide notifications after 5 seconds
  const notifications = document.querySelectorAll(".notification")

  notifications.forEach((notification) => {
    setTimeout(() => {
      notification.style.opacity = "0"
      setTimeout(() => {
        notification.remove()
      }, 500)
    }, 5000)
  })
})
