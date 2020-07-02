let mobilenet;

let vehiculoPrueba;

function modelReady(){
    console.log('Model is ready!!!');
    mobilenet.predict(vehiculoPrueba, gotResults);
}

function gotResults(error, results){
    if(error){
        console.error(error);
    } else {
        console.log(results);
        let label = results[0].className;
        let prob = results[0].probability;
        fill(0);
        textSize(10); //64
        text(label, 10, height - 100);
        createP(label);
        createP(prob);
    }

}

function imageReady(){
    image(vehiculoPrueba, 0, 0, width, height);
}

function setup(){
    createCanvas(640, 480);
    vehiculoPrueba = createImg('00001.jpg', imageReady);
    background(0);
    mobilenet = ml5.imageClassifier('Mobilenet', modelReady);

}