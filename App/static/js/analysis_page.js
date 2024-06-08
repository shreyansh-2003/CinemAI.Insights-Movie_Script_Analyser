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

function toggleMode() {
    const scriptButton = document.getElementById('script-button');
    const scriptContainer = document.getElementById('script-container');
    const metadataTable = document.querySelector('.meta table');

    // Check the current state of the button
    if (scriptButton.innerText === 'SCRIPT') {
        // If the button says "SCRIPT", show the script content and hide the table
        scriptContainer.style.display = 'block';
        metadataTable.style.display = 'none';
        scriptButton.innerText = 'METADATA';  // Update the button text
    } else {
        // If the button says "METADATA", show the table and hide the script content
        scriptContainer.style.display = 'none';
        metadataTable.style.display = 'block';
        scriptButton.innerText = 'SCRIPT';  // Update the button text
    }
}

// Function to show metadata
function showMetadata() {
    const scriptButton = document.getElementById('script-button');
    const scriptContainer = document.getElementById('script-container');
    const metadataTable = document.querySelector('.meta table');

    // Show the table and hide the script container
    scriptContainer.style.display = 'none';
    metadataTable.style.display = 'block';

    // Set the button text to "SCRIPT"
    scriptButton.innerText = 'SCRIPT';
}

// Function to show script
function showScript() {
    const scriptButton = document.getElementById('script-button');
    const scriptContainer = document.getElementById('script-container');
    const metadataTable = document.querySelector('.meta table');

    // Show the script container and hide the table
    scriptContainer.style.display = 'block';
    metadataTable.style.display = 'none';

    // Set the button text to "METADATA"
    scriptButton.innerText = 'METADATA';
}
