// class Main {

//     startWebcam() {
//     navigator.mediaDevices.getUserMedia({
//     video: {
//         facingMode: 'user'
//     },
//     audio: false
//     })
//     .then((stream) => {
//     this.video.srcObject = stream;
//     this.video.width = IMAGE_SIZE;
//     this.video.height = IMAGE_SIZE;
//     this.video.addEventListener('playing', () => this.videoPlaying = true);
//     this.video.addEventListener('paused', () => this.videoPlaying = false);
//     })
// }
// }

const IMAGE_SIZE = 227; // image size for the kNN model

class Main {
    constructor() {
        this.video = document.getElementById("video");
        this.canvas = document.getElementById("canvas");
        this.output = document.getElementById("output");
        this.videoPlaying = false;

        this.startWebcam();

        // Capture frame button
        document.getElementById("captureBtn").addEventListener('click', () => {
            this.captureFrame();
        });
    }

    // Start the webcam
    startWebcam() {
        navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false })
            .then((stream) => {
                this.video.srcObject = stream;
                this.video.addEventListener('playing', () => this.videoPlaying = true);
                this.video.addEventListener('paused', () => this.videoPlaying = false);
            })
            .catch(err => console.error("Error accessing webcam:", err));
    }

    // Capture a frame and extract features
    captureFrame() {
        if (!this.videoPlaying) {
            alert("Video not ready yet!");
            return;
        }

        const ctx = this.canvas.getContext('2d');
        // Draw the current video frame to the canvas, resizing it to IMAGE_SIZE x IMAGE_SIZE
        ctx.drawImage(this.video, 0, 0, IMAGE_SIZE, IMAGE_SIZE);

        // Convert the canvas to a Tensor
        const imageTensor = tf.browser.fromPixels(this.canvas)
            .expandDims(0) // add batch dimension
            .toFloat()
            .div(tf.scalar(255)); // normalize to [0,1]

        // For demonstration, flatten the tensor into a feature vector
        const featureVector = imageTensor.flatten();
        featureVector.data().then(data => {
            this.output.innerText = `Feature vector length: ${data.length}\nFirst 20 values:\n${Array.from(data).slice(0,20).map(v => v.toFixed(3)).join(', ')}`;
        });

        // Clean up tensor memory
        imageTensor.dispose();
    }
}

// Initialize Main
window.addEventListener('load', () => {
    const main = new Main();
});