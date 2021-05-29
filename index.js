import * as tf from '@tensorflow/tfjs';
import imagenet from './imagenet_labels.json';

//incarcam modelul
function loadMobilenet() {
	return tf.loadModel(
		'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json'
	);
}

function loadImage(src) {
	return new Promise((resolve, reject) => {
		const img = new Image();
		img.src = src;
		img.onload = () => resolve(tf.fromPixels(img));
		img.onerror = (err) => reject(err);
	});
}

function cropImage(img) {
	const width = img.shape[0];
	const height = img.shape[1];

	// minimul dintre lungime si latime
	const shorterSide = Math.min(img.shape[0], img.shape[1]);

	// calculam inceputul si sfarsitul punctelor pentru crop
	const startingHeight = (height - shorterSide) / 2;
	const startingWidth = (width - shorterSide) / 2;
	const endingHeight = startingHeight + shorterSide;
	const endingWidth = startingWidth + shorterSide;

	return img.slice(
		[startingWidth, startingHeight, 0],
		[endingWidth, endingHeight, 3]
	);
}

function resizeImage(image) {
	return tf.image.resizeBilinear(image, [224, 224]);
}

function batchImage(image) {
	//adaugam inca o dimensiune additionala, de marime 1
	const batchedImage = image.expandDims(0);
	batchedImage.norm().print();
	//transorma valoarea pixelului intr-un float intre -1 si 1
	//toFloat - conversia la float
	//div - imparte 2 tensori
	//scalar -> facem un scalar
	//sub -> subctracts - A - B;
	return batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
}

function loadAndProcessImage(image) {
	const croppedImage = cropImage(image);
	const resizedImage = resizeImage(croppedImage);
	const batchedImage = batchImage(resizedImage);
	return batchedImage;
}

loadMobilenet().then((pretrainedModel) => {
	image.onchange = () => {
		const [file] = image.files;
		if (file) {
			picture.src = URL.createObjectURL(file);
			loadImage(picture.src).then((img) => {
				const processedImage = loadAndProcessImage(img);
				const prediction = pretrainedModel.predict(processedImage);

				prediction.print();
				prediction.as1D().argMax().print();

				let out = prediction.as1D().argMax();
				let values = out.dataSync();
				let arr = Array.from(values);
				console.log(values);
				console.log(arr);
				console.log(imagenet[arr[0]]);

				document.getElementById('prediction').innerHTML = imagenet[arr[0]];
			});
		}
	};
});
// loadMobilenet().then((mobilenet) => {
// 	mobilenet.summary();
// });
