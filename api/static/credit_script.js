const form = document.getElementById("creditForm");
const submitBtn = document.getElementById("submitBtn");
const btnLoader = document.getElementById("btnLoader");
const resultContainer = document.getElementById("resultContainer");
const resultCard = document.getElementById("resultCard");
const riskLevel = document.getElementById("riskLevel");
const riskIndicator = document.getElementById("riskIndicator");
const criterionBadge = document.getElementById("criterionBadge");
const resultDetails = document.getElementById("resultDetails");
const errorMessage = document.getElementById("errorMessage");

// Risk level configurations
const riskConfig = {
  low: {
    class: "low",
    icon: "✓",
    color: "success",
    description: "Low risk client with favorable financial profile",
  },
  medium: {
    class: "medium",
    icon: "⚠",
    color: "warning",
    description: "Moderate risk requiring careful evaluation",
  },
  high: {
    class: "high",
    icon: "✗",
    color: "danger",
    description: "High risk client with concerning financial indicators",
  },
};

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

// Get risk configuration (case-insensitive)
function getRiskConfig(riskLevelStr) {
  const normalized = riskLevelStr.toLowerCase();
  return (
    riskConfig[normalized] || riskConfig.medium // Default to medium if unknown
  );
}

// Display prediction result
function displayResult(data) {
  const risk = data.risk_level;
  const criterion = data.model_criterion || "Decision Tree";
  const explanation = data.explanation || "";

  const config = getRiskConfig(risk);

  // Update risk level display
  riskLevel.textContent = risk;
  riskLevel.className = `risk-level ${config.class}`;

  // Update risk indicator
  riskIndicator.textContent = config.icon;
  riskIndicator.className = `risk-indicator ${config.class}`;

  // Update criterion badge
  criterionBadge.textContent = criterion;

  // Update details
  resultDetails.innerHTML = `
        <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(245, 87, 108, 0.2);">
            <p><strong>Model Method:</strong> ${criterion}</p>
            <p style="margin-top: 0.75rem; padding: 1rem; background: rgba(245, 87, 108, 0.05); border-radius: 8px; border-left: 3px solid rgba(245, 87, 108, 0.5);">
                ${config.description}
            </p>
            ${
              explanation
                ? `<p style="margin-top: 0.5rem; font-size: 0.85rem; opacity: 0.8;">${explanation}</p>`
                : ""
            }
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
  const clientData = {
    age: parseFloat(formData.get("age")),
    monthly_income: parseFloat(formData.get("monthly_income")),
    credit_history_years: parseFloat(formData.get("credit_history_years")),
    debt_ratio: parseFloat(formData.get("debt_ratio")),
    job_stability_years: parseFloat(formData.get("job_stability_years")),
  };

  // Validate all fields
  const fields = Object.keys(clientData);
  for (const field of fields) {
    const value = clientData[field];
    if (isNaN(value) || value === null || value === undefined) {
      showError(`Please fill in all fields with valid numbers`);
      return;
    }
  }

  // Validate age range
  if (clientData.age < 18 || clientData.age > 100) {
    showError("Age must be between 18 and 100 years");
    return;
  }

  // Validate monthly income
  if (clientData.monthly_income < 0) {
    showError("Monthly income must be greater than or equal to 0");
    return;
  }

  // Validate credit history years
  if (clientData.credit_history_years < 0) {
    showError("Credit history years must be greater than or equal to 0");
    return;
  }

  // Validate job stability years
  if (clientData.job_stability_years < 0) {
    showError("Job stability years must be greater than or equal to 0");
    return;
  }

  setLoading(true);

  try {
    const response = await fetch("/predict-risk", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(clientData),
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
        "Failed to get risk assessment. Please check your connection and try again."
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
