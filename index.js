import * as tf from '@tensorflow/tfjs';
import phone1 from './data/pexel-images/training/mobile/phone1.jpg';
import phone2 from './data/pexel-images/training/mobile/phone2.jpg';
import phone3 from './data/pexel-images/training/mobile/phone3.jpg';
import phone4 from './data/pexel-images/training/mobile/phone4.jpg';
import phone5 from './data/pexel-images/training/mobile/phone5.jpg';
import phone6 from './data/pexel-images/training/mobile/phone6.jpg';
import phone7 from './data/pexel-images/training/mobile/phone7.jpg';
import phone8 from './data/pexel-images/training/mobile/phone8.jpg';
import phone9 from './data/pexel-images/training/mobile/phone9.jpg';
import phone10 from './data/pexel-images/training/mobile/phone10.jpg';
import note1 from './data/pexel-images/training/notebook/note1.jpg';
import note2 from './data/pexel-images/training/notebook/note2.jpg';
import note3 from './data/pexel-images/training/notebook/note3.jpg';
import note4 from './data/pexel-images/training/notebook/note4.jpg';
import note5 from './data/pexel-images/training/notebook/note5.jpg';
import note6 from './data/pexel-images/training/notebook/note6.jpg';
import note7 from './data/pexel-images/training/notebook/note7.jpg';
import note8 from './data/pexel-images/training/notebook/note8.jpg';
import note9 from './data/pexel-images/training/notebook/note9.jpg';
import note10 from './data/pexel-images/training/notebook/note10.jpg';
import note_val1 from './data/pexel-images/validation/notebook/note_val1.jpg';
import phone_val1 from './data/pexel-images/validation/mobile/phone_val1.jpg';

// import note_val2 from './data/pexel-images/validation/notebook/note_val2.jpg';

const training = [
	phone1,
	phone2,
	phone3,
	phone4,
	phone5,
	phone6,
	phone7,
	phone8,
	phone9,
	phone10,
	note1,
	note2,
	note3,
	note4,
	note5,
	note6,
	note7,
	note8,
	note9,
	note10,
];

const labels = [
	'phone',
	'phone',
	'phone',
	'phone',
	'phone',
	'phone',
	'phone',
	'phone',
	'phone',
	'phone',
	'note',
	'note',
	'note',
	'note',
	'note',
	'note',
	'note',
	'note',
	'note',
	'note',
];

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

	// use the shorter side as the size to which we will crop
	const shorterSide = Math.min(img.shape[0], img.shape[1]);

	// calculate beginning and ending crop points
	const startingHeight = (height - shorterSide) / 2;
	const startingWidth = (width - shorterSide) / 2;
	const endingHeight = startingHeight + shorterSide;
	const endingWidth = startingWidth + shorterSide;

	// return image data cropped to those points
	return img.slice(
		[startingWidth, startingHeight, 0],
		[endingWidth, endingHeight, 3]
	);
}

function resizeImage(image) {
	return tf.image.resizeBilinear(image, [224, 224]);
}

function batchImage(image) {
	// Expand our tensor to have an additional dimension, whose size is 1
	const batchedImage = image.expandDims(0);

	// Turn pixel data into a float between -1 and 1.
	return batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
}

function loadAndProcessImage(image) {
	const croppedImage = cropImage(image);
	const resizedImage = resizeImage(croppedImage);
	const batchedImage = batchImage(resizedImage);
	return batchedImage;
}

// loadMobilenet().then((pretrainedModel) => {
// 	loadImage(chitara).then((img) => {
// 		const processedImage = loadAndProcessImage(img);
// 		const prediction = pretrainedModel.predict(processedImage);

// 		// Because of the way Tensorflow.js works, you must call print on a Tensor instead of console.log.
// 		prediction.print();
// 		prediction.as1D().argMax().print();
// 	});
// });

// loadMobilenet().then((mobilenet) => {
// 	mobilenet.summary();
// });

function buildPretrainedModel() {
	return loadMobilenet().then((mobilenet) => {
		const layer = mobilenet.getLayer('conv_pw_13_relu');
		return tf.model({
			inputs: mobilenet.inputs,
			outputs: layer.output,
		});
	});
}

function loadImages(images, pretrainedModel) {
	let promise = Promise.resolve();
	for (let i = 0; i < images.length; i++) {
		const image = images[i];
		promise = promise.then((data) => {
			return loadImage(image).then((loadedImage) => {
				// Note the use of `tf.tidy` and `.dispose()`. These are two memory management
				// functions that Tensorflow.js exposes.
				// https://js.tensorflow.org/tutorials/core-concepts.html
				//
				// Handling memory management is crucial for building a performant machine learning
				// model in a browser.
				return tf.tidy(() => {
					const processedImage = loadAndProcessImage(loadedImage);
					const prediction = pretrainedModel.predict(processedImage);

					if (data) {
						const newData = data.concat(prediction);
						data.dispose();
						return newData;
					}

					return tf.keep(prediction);
				});
			});
		});
	}

	return promise;
}

buildPretrainedModel().then((pretrainedModel) => {
	loadImages(training, pretrainedModel).then((xs) => {
		xs.print();
	});
});

function oneHot(labelIndex, classLength) {
	return tf.tidy(() =>
		tf.oneHot(tf.tensor1d([labelIndex]).toInt(), classLength)
	);
}

function getLabelsAsObject(labels) {
	let labelObject = {};
	for (let i = 0; i < labels.length; i++) {
		const label = labels[i];
		if (labelObject[label] === undefined) {
			// only assign it if we haven't seen it before
			labelObject[label] = Object.keys(labelObject).length;
		}
	}
	return labelObject;
}

function addLabels(labels) {
	return tf.tidy(() => {
		const classes = getLabelsAsObject(labels);
		const classLength = Object.keys(classes).length;

		let ys;
		for (let i = 0; i < labels.length; i++) {
			const label = labels[i];
			const labelIndex = classes[label];
			const y = oneHot(labelIndex, classLength);
			if (i === 0) {
				ys = y;
			} else {
				ys = ys.concat(y, 0);
			}
		}
		return ys;
	});
}

function getModel(numberOfClasses) {
	const model = tf.sequential({
		layers: [
			tf.layers.flatten({ inputShape: [7, 7, 256] }),
			tf.layers.dense({
				units: 100,
				activation: 'relu',
				kernelInitializer: 'varianceScaling',
				useBias: true,
			}),
			tf.layers.dense({
				units: numberOfClasses,
				kernelInitializer: 'varianceScaling',
				useBias: false,
				activation: 'softmax',
			}),
		],
	});

	model.compile({
		optimizer: tf.train.adam(0.0001),
		loss: 'categoricalCrossentropy',
		metrics: ['accuracy'],
	});

	return model;
}

function makePrediction(pretrainedModel, image, expectedLabel, model) {
	loadImage(image)
		.then((loadedImage) => {
			return loadAndProcessImage(loadedImage);
		})
		.then((loadedImage) => {
			const activatedImage = pretrainedModel.predict(loadedImage);
			loadedImage.dispose();
			return activatedImage;
		})
		.then((activatedImage) => {
			const prediction = model.predict(activatedImage);
			const predictionLabel = prediction.as1D().argMax().dataSync()[0];

			console.log('Expected Label', expectedLabel);
			console.log('Predicted Label', predictionLabel);

			prediction.dispose();
			activatedImage.dispose();
		});
}

buildPretrainedModel().then((pretrainedModel) => {
	loadImages(training, pretrainedModel).then((xs) => {
		const ys = addLabels(labels);

		const model = getModel(2);

		model
			.fit(xs, ys, {
				epochs: 20,
				shuffle: true,
			})
			.then((history) => {
				// make predictions
				makePrediction(pretrainedModel, phone_val1, '0', model);
				makePrediction(pretrainedModel, note_val1, '1', model);
			});
	});
});
