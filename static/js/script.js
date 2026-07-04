/**
 * AI-Based Spam Email Detector - JavaScript Logic
 * Handles client-side API requests to predict spam and manages form logic.
 */

document.addEventListener("DOMContentLoaded", () => {
    // --- SPAM DETECTOR PAGE LOGIC ---
    const detectorForm = document.getElementById("detector-form");
    const messageInput = document.getElementById("message-input");
    const analyzeBtn = document.getElementById("analyze-btn");
    const clearBtn = document.getElementById("clear-btn");
    const resultBox = document.getElementById("result-box");
    const resultLabel = document.getElementById("result-label");
    const confidenceText = document.getElementById("confidence-text");

    if (detectorForm) {
        // Handle Analyze Button Click
        detectorForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            const text = messageInput.value.trim();

            if (!text) {
                alert("Please enter some text or email content first.");
                return;
            }

            // Update UI to show loading state
            analyzeBtn.disabled = true;
            analyzeBtn.textContent = "Analyzing...";
            resultBox.style.display = "none";
            resultBox.className = "result-box"; // reset classes

            try {
                const response = await fetch("/predict", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify({ text: text }),
                });

                const data = await response.json();

                if (response.ok) {
                    // Update result elements
                    resultLabel.textContent = data.prediction;
                    confidenceText.textContent = `${data.confidence}%`;

                    // Set appropriate class based on classification label
                    if (data.prediction === "Spam") {
                        resultBox.classList.add("spam");
                    } else {
                        resultBox.classList.add("ham");
                    }
                    resultBox.style.display = "block";
                } else {
                    alert(data.error || "An error occurred during classification.");
                }
            } catch (error) {
                console.error("Error:", error);
                alert("Could not connect to the backend server. Please make sure the Flask app is running.");
            } finally {
                analyzeBtn.disabled = false;
                analyzeBtn.textContent = "Analyze";
            }
        });

        // Handle Clear Button Click
        clearBtn.addEventListener("click", () => {
            messageInput.value = "";
            resultBox.style.display = "none";
            resultBox.className = "result-box";
            messageInput.focus();
        });
    }

    // --- CONTACT PAGE LOGIC ---
    const contactForm = document.getElementById("contact-form");
    const successAlert = document.getElementById("success-alert");

    if (contactForm) {
        contactForm.addEventListener("submit", (e) => {
            e.preventDefault();

            // Simulate form submission
            successAlert.style.display = "block";
            contactForm.reset();

            // Auto-hide alert after 5 seconds
            setTimeout(() => {
                successAlert.style.display = "none";
            }, 5000);
        });
    }
});
