class Main {

    startWebcam() {
    navigator.mediaDevices.getUserMedia({
    video: {
        facingMode: 'user'
    },
    audio: false
    })
    .then((stream) => {
    this.video.srcObject = stream;
    this.video.width = IMAGE_SIZE;
    this.video.height = IMAGE_SIZE;
    this.video.addEventListener('playing', () => this.videoPlaying = true);
    this.video.addEventListener('paused', () => this.videoPlaying = false);
    })
}
}
