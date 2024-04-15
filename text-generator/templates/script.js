document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('prompt-form');
    const responseContainer = document.getElementById('response-container');

    form.addEventListener('submit', function(event) {
        event.preventDefault();
        const prompt = document.getElementById('prompt').value;
        console.log('Prompt:', prompt);
        generateResponse(prompt);
    });

    async function generateResponse(prompt) {
        try {
            const response = await fetch('/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ prompt: prompt })
            });
            const data = await response.json();
            console.log('Response:', data);
            displayResponse(data.response);
        } catch (error) {
            console.error('Error:', error);
        }
    }

    function displayResponse(response) {
        responseContainer.textContent = response;
    }
});
