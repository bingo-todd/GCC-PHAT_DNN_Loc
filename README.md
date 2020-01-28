## GCC-PHAT based DNN localization method
baseline system in *END-TO-END BINAURAL SOUND LOCALISATION FROM THE RAW WAVEFORM*[^Vecchiotti_2019]

## Framework

<img src='images/model/framework-gcc-phat.png'>

## Dataset
  Binarual signal are synthesized using BRIRs.

  - BRIRs

    Surrey binaural room impulse response (BRIR) database, including anechoic room and 4 reverberation room.
    <img src='images/dataset/rt-of-brir-dataset.png'>

  - Sound source

    TIMIT sentences

    Sentences per azimuth
    <table style='text-align:center'>
    <col width=15%>
    <col width=15%>
    <col width=15%>
      <tr>
        <td>Train</td> <td>Validate</td> <td>Evaluate</td>
      </tr>
      <tr>
        <td>24</td> <td>6</td> <td>15</td>
      </tr>
    </table>

## Cue extractor

  Normally, features are normalized before being fed into network.  If each dimension of features is independent variable, then normalization is applied to each dimension separately. For GCC-PHAT, what matters is the peak position, in other words, the relative value of each dimension, the same normalization coefficient should be used.

  Two types of normalization are tested here:
  -  **separate_norm**: each dimension is normalized separately
  -  **overall_norm**: all dimensions are normalized with the same factor

  E.g.
  | separate_norm  | overall_norm |
  |-|-|
  | <img src='images/dataset/separate_norm_example.png'> | <img src='images/dataset/overall_norm_example.png'> |

## Model training

### Multiconditional training(MCT)

  Each time, 1 reverberant room was selected in turn and using in evaluation, the other 3 reverberant rooms and the anechoic room were used in model training.
  <table>
    <tr>
      <th>separate_norm</th> <th>overall_norm</th>
    </tr>
    <tr>
      <th> <img src='images/training/train_process_mct_37dnorm.png'> </th> <th> <img src='images/training/train_process_mct_1dnorm.png'> </th>
    </tr>
  </table>


### Evaluation

  Localization result was reported every 25 frames, considering the existence of silent frames. The RMSE of sound azimuth is used as performance metrics. For more stable result, evaluation is ran on 4 different test sets and RMSEs are averaged (not in the ref. paper).

   <div align=center>
    <table style="text-align:center">
      <col width=20%>
      <col width=20%>
      <col width=20%>
      <col width=20%>
      <col width=20%>
      <thead>
        <tr>
          <th></th>
          <th>A</th>
          <th>B</th>
          <th>C</th>
          <th>D</th>
        </tr>
      </thead>
    <tbody>
      <tr>
        <td> Paper </td><td>2.7</td><td>3.3</td><td>3.1</td><td>5.2</td>
      </tr>
      <tr>
        <td>Separate_norm</td><td><strong>0.9</strong></td><td><strong>1.2</strong></td><td><strong>1.6</strong></td><td><strong>3.1</strong></td>
      </tr>
      <tr>
        <td>overall_norm</td><td>1.1</td><td>1.4</td><td>1.8</td><td>3.2</td>
      </tr>
    </tbody>
    </table>
    </div>

  **Separate_norm actually outperform overall_norm**, which is not expected.

## Reference
[^Vecchiotti_2019]: Vecchiotti, Paolo, Ning Ma, Stefano Squartini, and Guy J. Brown. “END-TO-END BINAURAL SOUND LOCALISATION FROM THE RAW WAVEFORM.” In 2019 IEEE INTERNATIONAL CONFERENCE ON ACOUSTICS, SPEECH AND SIGNAL PROCESSING (ICASSP), 451–55. International Conference on Acoustics Speech and Signal Processing ICASSP. 345 E 47TH ST, NEW YORK, NY 10017 USA: IEEE, 2019.
