document.addEventListener('DOMContentLoaded', function() {
    const canvas = document.getElementById('drawingCanvas');
    const ctx = canvas.getContext('2d');
    const clearBtn = document.getElementById('clearBtn');
    const calculateBtn = document.getElementById('calculateBtn');
    const expressionOutput = document.getElementById('expressionOutput');
    const resultOutput = document.getElementById('resultOutput');
    const canvasOverlay = document.querySelector('.canvas-overlay');
    
    // Canvas setup
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 8;  // Bolder strokes
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    
    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;
    
    // Get accurate canvas coordinates
    function getCanvasCoordinates(event) {
        const rect = canvas.getBoundingClientRect();
        return {
            x: (event.clientX - rect.left) * (canvas.width / rect.width),
            y: (event.clientY - rect.top) * (canvas.height / rect.height)
        };
    }
    
    // Touch support
    function handleTouch(e) {
        e.preventDefault();
        const touch = e.touches[0];
        const mouseEvent = new MouseEvent(
            e.type === 'touchstart' ? 'mousedown' : 'mousemove',
            {
                clientX: touch.clientX,
                clientY: touch.clientY
            }
        );
        canvas.dispatchEvent(mouseEvent);
    }
    
    // Event listeners
    canvas.addEventListener('mousedown', (e) => {
        isDrawing = true;
        const pos = getCanvasCoordinates(e);
        [lastX, lastY] = [pos.x, pos.y];
        canvasOverlay.style.display = 'none';
    });
    
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);
    canvas.addEventListener('touchstart', handleTouch);
    canvas.addEventListener('touchmove', handleTouch);
    canvas.addEventListener('touchend', stopDrawing);
    
    clearBtn.addEventListener('click', clearCanvas);
    calculateBtn.addEventListener('click', calculateExpression);
    
    function draw(e) {
        if (!isDrawing) return;
        
        const pos = getCanvasCoordinates(e);
        
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
        [lastX, lastY] = [pos.x, pos.y];
    }
    
    function stopDrawing() {
        isDrawing = false;
    }
    
    function clearCanvas() {
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        expressionOutput.textContent = '-';
        resultOutput.textContent = '-';
        canvasOverlay.style.display = 'block';
    }
    
    function calculateExpression() {
        const imageData = canvas.toDataURL('image/png');
        
        fetch(imageData)
            .then(res => res.blob())
            .then(blob => {
                const formData = new FormData();
                formData.append('file', blob, 'expression.png');
                return fetch('/', { method: 'POST', body: formData });
            })
            .then(response => response.json())
            .then(data => {
                expressionOutput.textContent = data.expression || 'Unrecognized';
                resultOutput.textContent = data.result || 'Invalid';
            })
            .catch(error => {
                console.error('Error:', error);
                expressionOutput.textContent = 'Processing error';
                resultOutput.textContent = '-';
            });
    }
});