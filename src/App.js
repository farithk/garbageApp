import React, { useState, useEffect } from 'react';

import './App.css';
import testImage from './glass70.jpg';

const tf = require('@tensorflow/tfjs');



function App() {

 
  const [valuePredcited, setValuePredicted] = useState(null);
  
  let GarbageType = ['Cardboard','Glass','Metal','Paper','Plastic','Trash']

  useEffect(() => {
    //console.log(valueToPredcit);
    run();
  },[]);
  
   
    const run = async () => {
        try {
          
          const model = await tf.loadLayersModel('http://farith.co/cnn/model.json');
          console.log(model);
          
          const img = document.getElementById('imageTest');
          const tfImg = tf.browser.fromPixels(img);
          const smalImg = tf.image.resizeBilinear(tfImg, [300, 300]);
          const resized = tf.cast(smalImg, 'float32');
          const t4d = tf.tensor4d(Array.from(resized.dataSync()),[1,300,300,3]).div(255)
          .sub([0.485, 0.456, 0.406])
          .div([0.229, 0.224, 0.225]);
          const prediction = model.predict(t4d);

          let maxEle = 0;
          let position = 0;

          for (let j = 0; j < prediction.dataSync().length; j++) {
            if(prediction.dataSync()[j] > maxEle){
              position = j;
              maxEle = prediction.dataSync()[position];
            }
          }
          console.log(smalImg.dataSync());
          console.log(prediction.dataSync());
          console.log(prediction.dataSync()[position]);

          let tpyeOfGarbage = GarbageType[position];
          setValuePredicted(tpyeOfGarbage);

        } catch (error) {
            console.log(error);
        }
    }

  return (
    <div className="App">
      <div className="container">
      
      <img src={testImage} id="imageTest" />
      <div className="subtitlePredicted">
        <p className="subtitlePredicted_inner">Garbage Category:<span className="subtitlePredicted_inner_result">{valuePredcited}</span></p> 
      </div>

      </div>
  
      
   
    </div>
  );
}

export default App;
