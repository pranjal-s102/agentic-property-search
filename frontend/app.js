const chatContainer = document.getElementById('chat-container');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
let sessionId = null;

// Event Listeners
sendBtn.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});

// Fetch initial greeting on page load
async function fetchInitialGreeting() {
    try {
        // Generate a new session ID for this conversation
        sessionId = crypto.randomUUID();

        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: "hello", session_id: sessionId })
        });

        if (!response.ok) throw new Error("Server error");

        const data = await response.json();
        sessionId = data.session_id;
        handleResponse(data.response);
    } catch (error) {
        console.error("Error fetching greeting:", error);
        appendMessage("G'day! 👋 I'm your Australian Real Estate Agent. How can I help you today?", 'ai');
    }
}

// Call on page load
fetchInitialGreeting();

async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    // Add User Message
    appendMessage(text, 'user');
    userInput.value = '';
    userInput.focus();

    // Show typing indicator (optional, skipping for MVP)

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text, session_id: sessionId })
        });

        if (!response.ok) throw new Error("Server error");

        const data = await response.json();
        sessionId = data.session_id;

        // Process Response
        handleResponse(data.response);

    } catch (error) {
        console.error("Error:", error);
        appendMessage("Sorry, I'm having trouble connecting to the server.", 'ai');
    }
}

function appendMessage(text, sender) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${sender}-message`;

    // Avatar
    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.innerHTML = sender === 'ai' ? '<i class="fa-solid fa-robot"></i>' : '<i class="fa-solid fa-user"></i>';

    // Content
    const content = document.createElement('div');
    content.className = 'content';

    if (sender === 'ai') {
        content.innerHTML = marked.parse(text);
    } else {
        content.innerText = text;
    }

    msgDiv.appendChild(avatar);
    msgDiv.appendChild(content);

    chatContainer.appendChild(msgDiv);
    scrollTop();
}

function handleResponse(rawText) {
    // Check for json_properties block
    // Regex matches ```json_properties [ ... ] ```
    const regex = /```json_properties\s*([\s\S]*?)\s*```/;
    const match = rawText.match(regex);

    let mainText = rawText;
    let properties = [];

    if (match) {
        try {
            properties = JSON.parse(match[1]);
            // Remove the JSON block from the display text
            mainText = rawText.replace(match[0], '');
        } catch (e) {
            console.error("JSON Parse error:", e);
        }
    }

    // Displays the text part
    appendMessage(mainText, 'ai');

    // If we have properties, render cards
    if (properties && properties.length > 0) {
        renderProperties(properties);
    }
}

function renderProperties(properties) {
    const gridDiv = document.createElement('div');
    gridDiv.className = 'property-grid';
    // Indent it to align with AI message content style? 
    // Or just append to chat container as a block

    // Let's wrap it in an AI message structure but without text, or just standalone
    // Standalone looks better like a "widget"

    const wrapper = document.createElement('div');
    wrapper.style.marginLeft = '56px'; // approximates avatar width + gap
    wrapper.style.marginBottom = '20px';
    wrapper.appendChild(gridDiv);

    properties.forEach(prop => {
        const card = document.createElement('div');
        card.className = 'property-card';

        // Image (Fallback if no image)
        const imgSrc = prop.image || prop.imgSrc || 'https://via.placeholder.com/400x300?text=No+Image';
        const price = prop.price || "Contact Agent";
        const title = prop.title || prop.address || "Property";
        const beds = prop.beds || prop.bedrooms || prop.features?.general?.bedrooms || '-';
        const baths = prop.baths || prop.bathrooms || prop.features?.general?.bathrooms || '-';
        const car = prop.carspaces || prop.features?.general?.parkingSpaces || '-';

        card.innerHTML = `
            <img src="${imgSrc}" class="property-image" alt="Property">
            <div class="property-details">
                <div class="property-price">${price}</div>
                <div class="property-title" title="${title}">${title}</div>
                <div class="property-meta">
                    <span><i class="fa-solid fa-bed"></i> ${beds}</span>
                    <span><i class="fa-solid fa-bath"></i> ${baths}</span>
                    <span><i class="fa-solid fa-car"></i> ${car}</span>
                </div>
            </div>
        `;

        // Optional clickable
        card.style.cursor = 'pointer';
        card.onclick = () => {
            const link = prop.link || (prop.listingId ? `https://www.realestate.com.au/${prop.listingId}` : 'https://www.realestate.com.au');
            window.open(link, '_blank');
        };

        gridDiv.appendChild(card);
    });

    chatContainer.appendChild(wrapper);
    scrollTop();
}

function scrollTop() {
    chatContainer.scrollTo({
        top: chatContainer.scrollHeight,
        behavior: 'smooth'
    });
}
