let model;
let img;

function results(err, res) {
	if (err) {
		console.error(err);
	} else {
		console.log(res);
		let out = res[0].label;
		let prob = res[0].confidence;
		fill(0);
		textSize(60);
		text(out, 10, height - 20);
		createP(out);
		createP(prob);
		// model.predict(results);
	}
}

function setup() {
	createCanvas(600, 400);
	img = createImg('img/cup.jpeg', () => {
		image(img, 0, 0, width, height);
	});
	// img = createCapture(VIDEO);
	img.hide();

	background(0);

	model = ml5.imageClassifier('MobileNet', () => {
		console.log('Model Ready');
		model.predict(img, results);
	});
}

// function draw() {
// 	image(img, 0, 0);
// }
