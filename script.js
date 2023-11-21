function login() {
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;

    // Simple validation (you should perform server-side validation in a real application)
    if (username && password) {
        alert(`Welcome, ${username}!`);
    } else {
        alert('Please enter both username and password.');
    }  
}
