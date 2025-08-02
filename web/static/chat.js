const form = document.getElementById('chat-form');
const log = document.getElementById('chat-log');

form.addEventListener('submit', async e => {
  e.preventDefault();
  const input = form.elements['prompt'];
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  appendMessage('user', text);
  const res = await fetch('/ask', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ prompt: text })
  });
  const { response } = await res.json();
  appendMessage('agent', response);
});

function appendMessage(who, text) {
  const div = document.createElement('div');
  div.className = `message ${who}`;
  div.textContent = text;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}
