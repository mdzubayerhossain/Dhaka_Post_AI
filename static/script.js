document.addEventListener('DOMContentLoaded', () => {
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');

    function autoResize() {
        this.style.height = 'auto';
        this.style.height = `${this.scrollHeight}px`;
    }

    userInput.addEventListener('input', autoResize);

    function addMessage(content, type) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', `${type}-message`);
        
        // Add typing animation for bot messages
        if (type === 'bot') {
            messageElement.innerHTML = '<span class="typing-indicator">•••</span>';
            chatMessages.appendChild(messageElement);
            
            setTimeout(() => {
                messageElement.innerHTML = content;
            }, 500);
        } else {
            messageElement.textContent = content;
            chatMessages.appendChild(messageElement);
        }

        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    async function sendMessage() {
        const query = userInput.value.trim();
        if (!query) return;

        addMessage(query, 'user');
        userInput.value = '';
        userInput.style.height = 'auto';

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query })
            });

            const data = await response.json();

            if (data.error) {
                addMessage(`Error: ${data.error}`, 'bot');
            } else {
                addMessage(data.response, 'bot');
            }
        } catch (error) {
            addMessage(`Network Error: ${error.message}`, 'bot');
        }
    }

    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
});