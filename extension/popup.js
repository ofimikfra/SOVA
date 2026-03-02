// Manual Toggle for the UI
document.getElementById('toggle-btn').addEventListener('click', () => {
    fetch('http://127.0.0.1:5000/manual_toggle');
})

// Existing Start Code
document.getElementById('start-btn').addEventListener('click', () => {
    fetch('http://127.0.0.1:5000/start');
});

document.querySelector('.primary').onclick = () => {
  fetch('http://127.0.0.1:5000/start')
    .then(response => response.json())
    .then(data => {
      console.log("Python script triggered!");
      alert("SOVA Assistance Starting...");
    })
    .catch(err => alert("Make sure your Python server is running!"));
};