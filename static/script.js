document.addEventListener('DOMContentLoaded', () => {
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');

    function addMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', `${sender}-message`);
        messageDiv.textContent = text;
        chatBox.appendChild(messageDiv);
        // Scroll to the bottom
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    async function sendMessage() {
        const messageText = userInput.value.trim();
        if (messageText === '') return;

        addMessage(messageText, 'user');
        userInput.value = ''; // Clear input field
        userInput.disabled = true; // Disable input while waiting for response
        sendButton.disabled = true;

        try {
            // Add a thinking indicator
            const thinkingDiv = document.createElement('div');
            thinkingDiv.classList.add('message', 'bot-message', 'thinking');
            thinkingDiv.textContent = '...';
            chatBox.appendChild(thinkingDiv);
            chatBox.scrollTop = chatBox.scrollHeight;


            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: messageText }),
            });

            // Remove thinking indicator
            chatBox.removeChild(thinkingDiv);

            if (!response.ok) {
                const errorData = await response.json();
                addMessage(`Error: ${errorData.error || 'Failed to get response'}`, 'bot');
                console.error('Error response from server:', errorData);
            } else {
                const data = await response.json();
                addMessage(data.response, 'bot');
            }
        } catch (error) {
             // Remove thinking indicator if it exists before showing error
            const thinking = chatBox.querySelector('.thinking');
            if (thinking) {
                chatBox.removeChild(thinking);
            }
            console.error('Error sending message:', error);
            addMessage('Error connecting to the chatbot server.', 'bot');
        } finally {
             userInput.disabled = false; // Re-enable input
             sendButton.disabled = false;
             userInput.focus(); // Focus back on input
        }
    }

    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            sendMessage();
        }
    });
});
