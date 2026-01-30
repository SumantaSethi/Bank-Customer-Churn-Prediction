// script.js
// JavaScript functionality for Bank Churn Prediction Application


// DOCUMENT READY

document.addEventListener('DOMContentLoaded', function() {
    console.log('Bank Churn Prediction App Loaded');
    
    // Initialize all features
    initFormValidation();
    initFormHelpers();
    initAnimations();
    initTooltips();
});


// FORM VALIDATION

function initFormValidation() {
    const form = document.getElementById('predictionForm');
    
    if (form) {
        form.addEventListener('submit', function(event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            
            form.classList.add('was-validated');
        }, false);
    }
}

// ============================================================================
// FORM HELPERS
// ============================================================================
function initFormHelpers() {
    // Credit Score Range Helper
    const creditScoreInput = document.getElementById('credit_score');
    if (creditScoreInput) {
        creditScoreInput.addEventListener('input', function() {
            const value = parseInt(this.value);
            const helpText = this.nextElementSibling;
            
            if (value >= 300 && value < 580) {
                helpText.textContent = '⚠️ Poor credit score';
                helpText.className = 'form-text text-danger';
            } else if (value >= 580 && value < 670) {
                helpText.textContent = '⚠️ Fair credit score';
                helpText.className = 'form-text text-warning';
            } else if (value >= 670 && value < 740) {
                helpText.textContent = '✓ Good credit score';
                helpText.className = 'form-text text-info';
            } else if (value >= 740 && value <= 850) {
                helpText.textContent = '✓ Excellent credit score';
                helpText.className = 'form-text text-success';
            }
        });
    }
    
    // Balance Formatter
    const balanceInput = document.getElementById('balance');
    if (balanceInput) {
        balanceInput.addEventListener('blur', function() {
            const value = parseFloat(this.value);
            if (!isNaN(value)) {
                this.value = value.toFixed(2);
            }
        });
    }
    
    // Salary Formatter
    const salaryInput = document.getElementById('estimated_salary');
    if (salaryInput) {
        salaryInput.addEventListener('blur', function() {
            const value = parseFloat(this.value);
            if (!isNaN(value)) {
                this.value = value.toFixed(2);
            }
        });
    }
    
    // Age Validation Helper
    const ageInput = document.getElementById('age');
    if (ageInput) {
        ageInput.addEventListener('input', function() {
            const value = parseInt(this.value);
            const helpText = this.nextElementSibling;
            
            if (value < 18) {
                helpText.textContent = '⚠️ Customer must be at least 18 years old';
                helpText.className = 'form-text text-danger';
            } else if (value > 100) {
                helpText.textContent = '⚠️ Please enter a valid age';
                helpText.className = 'form-text text-danger';
            } else {
                helpText.textContent = "Customer's age in years";
                helpText.className = 'form-text text-muted';
            }
        });
    }
}

// ============================================================================
// ANIMATIONS
// ============================================================================
function initAnimations() {
    // Fade in cards on scroll
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);
    
    // Observe all cards
    const cards = document.querySelectorAll('.card');
    cards.forEach(card => {
        observer.observe(card);
    });
}

// ============================================================================
// TOOLTIPS
// ============================================================================
function initTooltips() {
    // Initialize Bootstrap tooltips if available
    const tooltipTriggerList = [].slice.call(
        document.querySelectorAll('[data-bs-toggle="tooltip"]')
    );
    
    if (typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
        tooltipTriggerList.map(function(tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
}

// ============================================================================
// FORM RESET CONFIRMATION
// ============================================================================
document.addEventListener('DOMContentLoaded', function() {
    const resetBtn = document.querySelector('button[type="reset"]');
    
    if (resetBtn) {
        resetBtn.addEventListener('click', function(event) {
            if (!confirm('Are you sure you want to reset the form?')) {
                event.preventDefault();
            }
        });
    }
});

// ============================================================================
// SMOOTH SCROLL
// ============================================================================
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// ============================================================================
// FORM AUTO-SAVE (Optional - saves to localStorage)
// ============================================================================
function saveFormData() {
    const form = document.getElementById('predictionForm');
    
    if (form) {
        const formData = new FormData(form);
        const data = {};
        
        formData.forEach((value, key) => {
            data[key] = value;
        });
        
        localStorage.setItem('churnPredictionFormData', JSON.stringify(data));
        console.log('Form data saved');
    }
}

function loadFormData() {
    const savedData = localStorage.getItem('churnPredictionFormData');
    
    if (savedData) {
        const data = JSON.parse(savedData);
        const form = document.getElementById('predictionForm');
        
        if (form && confirm('Would you like to restore your previous form data?')) {
            Object.keys(data).forEach(key => {
                const input = form.elements[key];
                if (input) {
                    if (input.type === 'radio') {
                        const radioBtn = form.querySelector(
                            `input[name="${key}"][value="${data[key]}"]`
                        );
                        if (radioBtn) radioBtn.checked = true;
                    } else {
                        input.value = data[key];
                    }
                }
            });
        }
    }
}

// Auto-save form data on input
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictionForm');
    
    if (form) {
        // Load saved data
        loadFormData();
        
        // Save on input change
        form.addEventListener('change', saveFormData);
        
        // Clear saved data on successful submission
        form.addEventListener('submit', function() {
            localStorage.removeItem('churnPredictionFormData');
        });
    }
});

// ============================================================================
// NUMBER FORMATTING
// ============================================================================
function formatNumber(number, decimals = 2) {
    return number.toFixed(decimals).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

function formatCurrency(amount) {
    return '$' + formatNumber(amount, 2);
}

// ============================================================================
// LOADING INDICATOR
// ============================================================================
function showLoading(buttonElement) {
    const originalText = buttonElement.innerHTML;
    buttonElement.innerHTML = '<span class="loading"></span> Processing...';
    buttonElement.disabled = true;
    buttonElement.dataset.originalText = originalText;
}

function hideLoading(buttonElement) {
    buttonElement.innerHTML = buttonElement.dataset.originalText;
    buttonElement.disabled = false;
}

// Add loading state to form submission
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictionForm');
    
    if (form) {
        form.addEventListener('submit', function(event) {
            if (form.checkValidity()) {
                const submitBtn = form.querySelector('button[type="submit"]');
                if (submitBtn) {
                    showLoading(submitBtn);
                }
            }
        });
    }
});

// ============================================================================
// PRINT RESULTS
// ============================================================================
function printResults() {
    window.print();
}

// ============================================================================
// EXPORT RESULTS TO PDF (requires html2pdf library)
// ============================================================================
function exportToPDF() {
    const element = document.querySelector('.container');
    const opt = {
        margin: 1,
        filename: 'churn-prediction-results.pdf',
        image: { type: 'jpeg', quality: 0.98 },
        html2canvas: { scale: 2 },
        jsPDF: { unit: 'in', format: 'letter', orientation: 'portrait' }
    };
    
    if (typeof html2pdf !== 'undefined') {
        html2pdf().set(opt).from(element).save();
    } else {
        alert('PDF export library not loaded');
    }
}

// ============================================================================
// KEYBOARD SHORTCUTS
// ============================================================================
document.addEventListener('keydown', function(event) {
    // Ctrl/Cmd + Enter to submit form
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
        const form = document.getElementById('predictionForm');
        if (form && form.checkValidity()) {
            form.submit();
        }
    }
    
    // Ctrl/Cmd + R to reset form
    if ((event.ctrlKey || event.metaKey) && event.key === 'r') {
        event.preventDefault();
        const form = document.getElementById('predictionForm');
        if (form && confirm('Reset form?')) {
            form.reset();
        }
    }
});

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// ============================================================================
// CONSOLE WELCOME MESSAGE
// ============================================================================
console.log('%c Bank Churn Prediction System ', 
    'background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; font-size: 20px; padding: 10px;');
console.log('%c Powered by Machine Learning ', 
    'background: #333; color: #bada55; font-size: 12px; padding: 5px;');
console.log('');
console.log('💡 Tip: Press Ctrl+Enter to submit the form quickly!');
console.log('💡 Tip: Press Ctrl+R to reset the form!');
