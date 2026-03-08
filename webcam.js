const IMAGE_SIZE = 227; // for feature extraction

class Main {
    constructor() {
        this.video = document.getElementById("video");
        this.canvas = document.getElementById("canvas");
        this.output = document.getElementById("output");
        this.videoPlaying = false;

        this.startWebcam();

        document.getElementById("captureBtn").addEventListener('click', () => {
            this.captureFrame();
        });
    }

    startWebcam() {
        // Only start after the page is fully loaded
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            alert("Webcam not supported in this browser.");
            return;
        }

        navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false })
            .then((stream) => {
                this.video.srcObject = stream;
                this.video.addEventListener('loadedmetadata', () => {
                    this.video.play();
                    this.videoPlaying = true;
                });
            })
            .catch(err => {
                console.error("Error accessing webcam:", err);
                alert("Could not access webcam: " + err.message);
            });
    }

    captureFrame() {
        if (!this.videoPlaying) {
            alert("Video not ready yet!");
            return;
        }

        const ctx = this.canvas.getContext('2d');
        ctx.drawImage(this.video, 0, 0, IMAGE_SIZE, IMAGE_SIZE);

        // Convert canvas to Tensor
        const imageTensor = tf.browser.fromPixels(this.canvas)
            .expandDims(0)  // batch dimension
            .toFloat()
            .div(tf.scalar(255)); // normalize

        // Flatten to feature vector
        const featureVector = imageTensor.flatten();
        featureVector.data().then(data => {
            this.output.innerText = `Feature vector length: ${data.length}\nFirst 20 values:\n${Array.from(data).slice(0,20).map(v => v.toFixed(3)).join(', ')}`;
        });

        imageTensor.dispose();
    }
}

// Initialize after page fully loaded
window.addEventListener('load', () => {
    new Main();
});