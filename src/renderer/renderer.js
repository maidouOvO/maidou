const { ipcRenderer } = require('electron');

document.getElementById('processButton').addEventListener('click', async () => {
    const fileInput = document.getElementById('pdfInput');
    const file = fileInput.files[0];

    if (!file) {
        updateStatus('Please select a PDF file first', 'error');
        return;
    }

    const bgWidth = parseInt(document.getElementById('bgWidth').value);
    const bgHeight = parseInt(document.getElementById('bgHeight').value);
    const processingMode = document.getElementById('processingMode').value;

    if (isNaN(bgWidth) || isNaN(bgHeight) || bgWidth <= 0 || bgHeight <= 0) {
        updateStatus('Please enter valid background dimensions', 'error');
        return;
    }

    updateStatus('Processing PDF...', 'info');

    try {
        const buffer = await file.arrayBuffer();
        const result = await ipcRenderer.invoke('process-pdf', {
            pdfBuffer: Buffer.from(buffer),
            bgWidth,
            bgHeight,
            processingMode
        });

        if (result.success) {
            updateStatus('PDF processed successfully!', 'success');
        } else {
            updateStatus(`Error: ${result.error}`, 'error');
        }
    } catch (error) {
        updateStatus(`Error: ${error.message}`, 'error');
    }
});

function updateStatus(message, type) {
    const statusDiv = document.getElementById('status');
    statusDiv.textContent = message;
    statusDiv.style.backgroundColor = type === 'error' ? '#ffebee' :
                                    type === 'success' ? '#e8f5e9' :
                                    '#e3f2fd';
    statusDiv.style.color = type === 'error' ? '#c62828' :
                           type === 'success' ? '#2e7d32' :
                           '#1565c0';
}
