const canvas = document.getElementById('trackCanvas');
const ctx = canvas.getContext('2d');

let blueCones = [];
let yellowCones = [];
const coneRadius = 6;

function resizeCanvasToDisplaySize() {
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width;
  canvas.height = rect.height;
  redrawCones();
}

// Flip the canvas Y-axis to make (0,0) bottom-left
function setCanvasTransform() {
  ctx.setTransform(1, 0, 0, -1, 0, canvas.height);  // flip Y-axis
}

// Convert mouse click Y to flipped canvas Y
function getCanvasCoordinates(event) {
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = canvas.height - (event.clientY - rect.top);  // flip Y
  return { x, y };
}

function drawCone(x, y, color) {
  ctx.beginPath();
  ctx.arc(x, y, coneRadius, 0, 2 * Math.PI);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.strokeStyle = 'black';
  ctx.lineWidth = 1;
  ctx.stroke();
}

function redrawCones() {
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);  // reset to normal
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.restore();

  setCanvasTransform();  // flip Y-axis for bottom-left origin

  // Draw lines between blue cones
  if (blueCones.length >= 2) {
    ctx.beginPath();
    ctx.moveTo(blueCones[0].x, blueCones[0].y);
    for (let i = 1; i < blueCones.length; i++) {
      ctx.lineTo(blueCones[i].x, blueCones[i].y);
    }
    ctx.strokeStyle = 'blue';
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  // Draw lines between yellow cones
  if (yellowCones.length >= 2) {
    ctx.beginPath();
    ctx.moveTo(yellowCones[0].x, yellowCones[0].y);
    for (let i = 1; i < yellowCones.length; i++) {
      ctx.lineTo(yellowCones[i].x, yellowCones[i].y);
    }
    ctx.strokeStyle = 'goldenrod';
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  // Draw cones — first in orange, rest in color
  blueCones.forEach(({ x, y }, i) => {
    const color = i === 0 ? 'orangered' : 'blue';
    drawCone(x, y, color);
  });

  yellowCones.forEach(({ x, y }, i) => {
    const color = i === 0 ? 'orangered' : 'goldenrod';
    drawCone(x, y, color);
  });
}

canvas.addEventListener('contextmenu', e => e.preventDefault());

canvas.addEventListener('mousedown', e => {
  const { x, y } = getCanvasCoordinates(e);

  if (e.button === 0) {
    blueCones.push({ x, y });
    drawCone(x, y, 'blue');
  } else if (e.button === 2) {
    yellowCones.push({ x, y });
    drawCone(x, y, 'goldenrod');
  }

  redrawCones();
});

// Handle control buttons
document.querySelectorAll('.button').forEach(button => {
  button.addEventListener('click', () => {
    const label = button.textContent.toLowerCase();

    if (label.includes('clear last blue')) {
      blueCones.pop();
    } else if (label.includes('clear last yellow')) {
      yellowCones.pop();
    } else if (label.includes('clear all blue')) {
      blueCones = [];
    } else if (label.includes('clear all yellow')) {
      yellowCones = [];
    } else if (label.includes('download')) {
      downloadConesAsCSV();
    }

    redrawCones();
  });
});

// Export to CSV in 100x60 space
function downloadConesAsCSV() {
  const frameWidth = 100;
  const frameHeight = 60;

  const canvasWidth = canvas.width;
  const canvasHeight = canvas.height;

  function scale(x, y) {
    const scaledX = (x / canvasWidth) * frameWidth;
    const scaledY = (y / canvasHeight) * frameHeight;
    return [scaledX, scaledY];
  }

  let csv = "x,y,type\n";

  blueCones.forEach(({ x, y }) => {
    const [sx, sy] = scale(x, y);
    csv += `${sx.toFixed(3)},${sy.toFixed(3)},left\n`;
  });

  yellowCones.forEach(({ x, y }) => {
    const [sx, sy] = scale(x, y);
    csv += `${sx.toFixed(3)},${sy.toFixed(3)},right\n`;
  });

  const blob = new Blob([csv], { type: 'text/csv' });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = 'racetrack_cones_scaled.csv';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// Initialize
resizeCanvasToDisplaySize();
window.addEventListener('resize', () => {
  resizeCanvasToDisplaySize();
});
