// script.js

// Function to handle click on an image
// Function to handle click on an image
function handleImageClick(imdbId) {
    // Redirect to the new page with IMDb ID in the URL
    window.location.href = "/get_data/" + imdbId;
}

// Show lightbox when an element with the 'open' class is clicked
document.addEventListener('click', function (event) {
    if (event.target.matches('.open')) {
        // Get the IMDb ID from the clicked image
        var imdbId = event.target.parentElement.id;
        handleImageClick(imdbId);
    }
}, false);

// Handle keydown event (Enter key) to trigger click on active element
document.addEventListener(
    "keydown", (e) => {
        if (e.keyCode == 13) {
            document.activeElement.click();
            document.querySelector('body').classList.add('fixed');
        }
    }, false);

// Hide lightbox when an element with the 'close' class is clicked
document.addEventListener('click', function (event) {
    if (!event.target.matches('.close')) return;
    document.querySelector('body').classList.remove('fixed');
}, false);

// Handle keydown event (Esc key) to close the lightbox
document.addEventListener(
    "keydown", (e) => {
        if (e.keyCode == 27) {
            document.activeElement.querySelector('.close').click();
            document.querySelector('body').classList.remove('fixed');
        }
    }, false);

// ChatGPT Style Writing
const mainHeading = document.getElementById('main-heading');
const typewriterContainer = document.getElementById('typewriter-container');
const elementsToType = [mainHeading];
let index = 0;

const text = "CinemAI Insights - A One Stop Solution";

function type() {
    if (index < text.length) {
        elementsToType.forEach(element => {
            element.innerHTML = text.slice(0, index) + '<span class="blinking-cursor">|</span>';
        });

        index++;
        setTimeout(type, Math.random() * 120 + 30);
    } else {
        // Wait for a moment and then start erasing
        setTimeout(erase, 800);
    }
}

function erase() {
    if (index >= 0) {
        elementsToType.forEach(element => {
            element.innerHTML = text.slice(0, index) + '<span class="blinking-cursor">|</span>';
        });

        index--;
        setTimeout(erase, Math.random() * 120 + 30);
    } else {
        // Wait for a moment and then start typing again
        setTimeout(type, 1000);
    }
}

// Start typing
type();
