document.addEventListener("DOMContentLoaded", () => {
    initNeuralBackground();
    initTiltEffect();
    // File upload logic is handled inline in HTML or needs to be adapted here if we want to move it.
    // Given the HTML already has script tags for file handling, we'll keep this file focused on visuals
    // to avoid conflicts, or we can consolidate. Let's Consolidate for cleaner code.
});

/* ─── Neural Network Background ──────────────────────────────────────── */
function initNeuralBackground() {
    const canvas = document.getElementById("neural-canvas");
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    let width, height;
    let particles = [];
    
    // Config
    const PARTICLE_COUNT = 80;
    const CONNECT_DISTANCE = 150;
    const MOUSE_DISTANCE = 200;

    let mouse = { x: null, y: null };

    const resize = () => {
        width = canvas.width = window.innerWidth;
        height = canvas.height = window.innerHeight;
    };
    window.addEventListener("resize", resize);
    resize();

    window.addEventListener("mousemove", e => {
        mouse.x = e.x;
        mouse.y = e.y;
    });

    class Particle {
        constructor() {
            this.x = Math.random() * width;
            this.y = Math.random() * height;
            this.vx = (Math.random() - 0.5) * 0.5;
            this.vy = (Math.random() - 0.5) * 0.5;
            this.size = Math.random() * 2 + 1;
            this.color = "rgba(0, 242, 255, " + (Math.random() * 0.5 + 0.2) + ")"; 
        }

        update() {
            if (this.x > width || this.x < 0) this.vx = -this.vx;
            if (this.y > height || this.y < 0) this.vy = -this.vy;

            // Mouse interaction
            let dx = mouse.x - this.x;
            let dy = mouse.y - this.y;
            let distance = Math.sqrt(dx*dx + dy*dy);
            
            if (distance < MOUSE_DISTANCE) {
                const forceDirectionX = dx / distance;
                const forceDirectionY = dy / distance;
                const force = (MOUSE_DISTANCE - distance) / MOUSE_DISTANCE;
                const directionX = forceDirectionX * force * 2; // Stronger push
                const directionY = forceDirectionY * force * 2;
                this.vx -= directionX * 0.05; 
                this.vy -= directionY * 0.05;
            }

            this.x += this.vx;
            this.y += this.vy;
        }

        draw() {
            ctx.fillStyle = this.color;
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    function init() {
        particles = [];
        for (let i = 0; i < PARTICLE_COUNT; i++) {
            particles.push(new Particle());
        }
    }
    init();

    function animate() {
        requestAnimationFrame(animate);
         ctx.clearRect(0, 0, width, height);

        for (let i = 0; i < particles.length; i++) {
            particles[i].update();
            particles[i].draw();

            for (let j = i; j < particles.length; j++) {
                let dx = particles[i].x - particles[j].x;
                let dy = particles[i].y - particles[j].y;
                let distance = Math.sqrt(dx*dx + dy*dy);

                if (distance < CONNECT_DISTANCE) {
                    ctx.beginPath();
                    ctx.strokeStyle = "rgba(0, 242, 255," + (1 - distance/CONNECT_DISTANCE) * 0.15 + ")";
                    ctx.lineWidth = 1;
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.stroke();
                }
            }
        }
    }
    animate();
}

/* ─── 3D Tilt Effect ─────────────────────────────────────────────────── */
function initTiltEffect() {
    const cards = document.querySelectorAll(".tilt-card, .glass-panel");
    
    cards.forEach(card => {
        card.addEventListener("mousemove", e => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;
            
            // Limit tilt to small range so it does not look broken
            const rotateX = ((y - centerY) / centerY) * -5; 
            const rotateY = ((x - centerX) / centerX) * 5;

            card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) scale3d(1.02, 1.02, 1.02)`;
        });

        card.addEventListener("mouseleave", () => {
            card.style.transform = "perspective(1000px) rotateX(0) rotateY(0) scale3d(1, 1, 1)";
        });
    });
}
