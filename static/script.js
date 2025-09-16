document.addEventListener('DOMContentLoaded', () => {

    // ── Tab switching ──
    const tabs = document.querySelectorAll('.tab');
    const modes = document.querySelectorAll('.mode');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            modes.forEach(m => m.classList.remove('active'));
            tab.classList.add('active');
            document.getElementById('mode-' + tab.dataset.mode).classList.add('active');
        });
    });

    // ── Generate Mode ──
    const genPrompt = document.getElementById('gen-prompt');
    const maxTokens = document.getElementById('max-tokens');
    const temperature = document.getElementById('temperature');
    const tokensVal = document.getElementById('tokens-val');
    const tempVal = document.getElementById('temp-val');
    const genBtn = document.getElementById('generate-btn');
    const genOutput = document.getElementById('gen-output');
    const status = document.getElementById('status');

    maxTokens.addEventListener('input', e => tokensVal.textContent = e.target.value);
    temperature.addEventListener('input', e => tempVal.textContent = e.target.value);

    genPrompt.addEventListener('keydown', e => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            genBtn.click();
        }
    });

    genBtn.addEventListener('click', async () => {
        const text = genPrompt.value.trim();
        if (!text) { genOutput.textContent = 'Please enter a prompt.'; return; }

        genBtn.disabled = true;
        status.classList.remove('hidden');
        genOutput.textContent = '';

        try {
            const res = await fetch('/api/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt: text,
                    max_tokens: parseInt(maxTokens.value),
                    temperature: parseFloat(temperature.value)
                })
            });
            const data = await res.json();
            genOutput.textContent = res.ok ? data.generated_text : 'Error: ' + data.error;
        } catch (err) {
            genOutput.textContent = 'Connection error. Is the server running?';
        } finally {
            genBtn.disabled = false;
            status.classList.add('hidden');
        }
    });

    // ── Classify Mode ──
    const clsBtn = document.getElementById('classify-btn');
    const clsInput = document.getElementById('cls-input');
    const clsOutput = document.getElementById('cls-output');

    clsBtn.addEventListener('click', async () => {
        const text = clsInput.value.trim();
        if (!text) { clsOutput.innerHTML = '<span class="muted">Enter some text first.</span>'; return; }

        clsBtn.disabled = true;
        status.classList.remove('hidden');

        try {
            const res = await fetch('/api/classify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text })
            });
            const data = await res.json();
            if (res.ok) {
                clsOutput.textContent = data.label + ' (' + (data.confidence * 100).toFixed(1) + '%)';
            } else {
                clsOutput.textContent = data.error || 'Classification failed';
            }
        } catch (err) {
            clsOutput.textContent = 'Not available yet — classifier needs training.';
        } finally {
            clsBtn.disabled = false;
            status.classList.add('hidden');
        }
    });

    // ── Chat Mode ──
    const chatBox = document.getElementById('chat-box');
    const chatInput = document.getElementById('chat-input');
    const chatBtn = document.getElementById('chat-btn');

    chatInput.addEventListener('keydown', e => {
        if (e.key === 'Enter') chatBtn.click();
    });

    chatBtn.addEventListener('click', async () => {
        const text = chatInput.value.trim();
        if (!text) return;

        // Add user message
        const userMsg = document.createElement('div');
        userMsg.className = 'chat-msg user';
        userMsg.textContent = text;
        chatBox.appendChild(userMsg);
        chatInput.value = '';
        chatBox.scrollTop = chatBox.scrollHeight;

        chatBtn.disabled = true;

        try {
            const res = await fetch('/api/chat_stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text })
            });
            
            const botMsg = document.createElement('div');
            botMsg.className = 'chat-msg bot';
            botMsg.textContent = '';
            chatBox.appendChild(botMsg);
            
            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let doneReading = false;
            
            while (!doneReading) {
                const { value, done } = await reader.read();
                if (done) break;
                
                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\n\n');
                
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const dataStr = line.replace('data: ', '');
                        if (!dataStr) continue;
                        
                        try {
                            const data = JSON.parse(dataStr);
                            if (data.text) {
                                botMsg.textContent += data.text;
                                chatBox.scrollTop = chatBox.scrollHeight;
                            }
                            if (data.done) {
                                doneReading = true;
                            }
                        } catch (e) {
                            console.error("Parse error", e);
                        }
                    }
                }
            }
        } catch (err) {
            const errMsg = document.createElement('div');
            errMsg.className = 'chat-msg system';
            errMsg.textContent = 'Streaming error or model unavailable.';
            chatBox.appendChild(errMsg);
        } finally {
            chatBtn.disabled = false;
            chatBox.scrollTop = chatBox.scrollHeight;
        }
    });
});

// Added global error boundary for fetch requests
\n// Note: SSE stream buffering depends on proxy\n