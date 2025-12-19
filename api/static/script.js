const form = document.getElementById("pricingForm");
const submitBtn = document.getElementById("submitBtn");
const btnLoader = document.getElementById("btnLoader");
const resultContainer = document.getElementById("resultContainer");
const resultCard = document.getElementById("resultCard");
const priceValue = document.getElementById("priceValue");
const methodBadge = document.getElementById("methodBadge");
const resultDetails = document.getElementById("resultDetails");
const errorMessage = document.getElementById("errorMessage");

// Format number with commas
function formatNumber(num) {
  return new Intl.NumberFormat("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(num);
}

// Show error message
function showError(message) {
  errorMessage.textContent = message;
  errorMessage.classList.add("show");

  setTimeout(() => {
    errorMessage.classList.remove("show");
  }, 5000);
}

// Hide error message
function hideError() {
  errorMessage.classList.remove("show");
}

// Show loading state
function setLoading(loading) {
  if (loading) {
    submitBtn.classList.add("loading");
    submitBtn.disabled = true;
  } else {
    submitBtn.classList.remove("loading");
    submitBtn.disabled = false;
  }
}

// Display prediction result
function displayResult(data) {
  const price = data.predicted_price;
  const method = data.method || "Mini-Batch GD";

  priceValue.textContent = formatNumber(price);
  methodBadge.textContent = method;

  resultDetails.innerHTML = `
        <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(102, 126, 234, 0.2);">
            <p><strong>Model Method:</strong> ${method}</p>
            <p style="margin-top: 0.5rem; font-size: 0.85rem; opacity: 0.8;">
                This prediction is based on gradient descent optimization trained on historical pricing data.
            </p>
        </div>
    `;

  resultCard.classList.add("show");

  // Scroll to result
  resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// Handle form submission
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  hideError();

  // Collect form data
  const formData = new FormData(form);
  const features = [
    parseFloat(formData.get("demand_index")),
    parseFloat(formData.get("time_slot")),
    parseFloat(formData.get("day_of_week")),
    parseFloat(formData.get("competition_pressure")),
    parseFloat(formData.get("operational_cost")),
    parseFloat(formData.get("seasonality_index")),
    parseFloat(formData.get("marketing_intensity")),
  ];

  // Validate features
  if (features.some((f) => isNaN(f) || f === null || f === undefined)) {
    showError("Please fill in all fields with valid numbers");
    return;
  }

  // Validate demand_index >= 0
  if (features[0] < 0) {
    showError("Demand Index must be greater than or equal to 0");
    return;
  }

  // Validate day_of_week range
  if (features[2] < 0 || features[2] > 6) {
    showError("Day of Week must be between 0 and 6");
    return;
  }

  setLoading(true);

  try {
    const response = await fetch("/predict-price", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ features }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(
        errorData.error || `HTTP error! status: ${response.status}`
      );
    }

    const data = await response.json();
    displayResult(data);
  } catch (error) {
    console.error("Error:", error);
    showError(
      error.message ||
        "Failed to get prediction. Please check your connection and try again."
    );
    resultCard.classList.remove("show");
  } finally {
    setLoading(false);
  }
});

// Add input validation feedback
const inputs = form.querySelectorAll('input[type="number"]');
inputs.forEach((input) => {
  input.addEventListener("input", function () {
    // Remove any previous error styling
    this.style.borderColor = "";

    // Validate on blur
    this.addEventListener("blur", function () {
      if (this.validity.valid) {
        this.style.borderColor = "";
      } else {
        this.style.borderColor = "#f5576c";
      }
    });
  });
});

// Add smooth number input formatting
inputs.forEach((input) => {
  input.addEventListener("focus", function () {
    this.select();
  });
});
