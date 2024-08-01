[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/ol4GAg0d)
[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-2972f46106e565e64193e422d61a12cf1da4916b45550586e14ef0a7c637dd04.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=15423638)

<!-- Name of your teams' final project -->
<div align="center" style="background-color: white; padding: 10px;">
  <h1>Fine-tuning Automatic Speech Recognition (ASR) Models for Accented Speech</h1>
</div>

<table align="center" style="background-color: white; border-collapse: collapse;">
  <tr>
    <td align="center" style="padding: 10px; border: none;">
      <img src="https://mma.prnewswire.com/media/967123/NACME_Logo.jpg" alt="NACME Logo" width="200">
    </td>
    <td align="center" style="padding: 10px; border: none;">
      <img src="https://upload.wikimedia.org/wikipedia/commons/f/fa/Apple_logo_black.svg" alt="Apple Logo" width="115">
    </td>
    <td align="center" style="padding: 10px; border: none;">
      <img src="https://1000logos.net/wp-content/uploads/2020/10/University-of-Southern-California-logo.png" alt="USC Logo" width="400">
    </td>
  </tr>
</table>

## [National Action Council for Minorities in Engineering(NACME)](https://www.nacme.org) Artificial Intelligence - Machine Learning (AIML) Intensive Summer Bootcamp at the [University of Southern California](https://viterbischool.usc.edu)

<!-- List all of the members who developed the project and link to each members respective GitHub profile -->
Developed by: [Dialect Dynamics](https://github.com/orgs/NACME-AIML-2024/teams/dialect-dynamics)

*[Name] - `Major` - `Undergraduate Institution` - `Role`*

- [Sebastián A. Cruz Romero](https://github.com/romerocruzsa) - `Computer Science and Engineering` - `University of Puerto Rico at Mayagüez` - `Lead`
- [Aliya Daire](https://github.com/aliya-daire) - `Computer Science and Business Administration` - `University of Southern California` - `Notes`
- [Brandon Medina](https://github.com/bj-khaled) - `Electrical Engineering` - `Texas A&M - Kingsville`
- [Samuel Ovalle](https://github.com/edrop7) - `Mechanical Engineering` - `Florida International University` - `Notes`

## Description
<!-- Give a short description on what your project accomplishes and what tools it uses. In addition, you can drop screenshots directly into your README file to add them to your README. Take these from your presentations. -->

### Project Summary (PDF)
<table align="center" style="background-color: white; border-collapse: collapse;">
  <tr>
    <td align="center" style="padding: 10px; border: none;">
      <img src="https://github.com/user-attachments/assets/43231258-1ec4-4aec-afb0-a2e8b76ac270" alt="project-desc-p1" width="499">
    </td>
    <td align="center" style="padding: 10px; border: none;">
      <img src="https://github.com/user-attachments/assets/b4264e13-d3b8-4d29-ad5b-c0580dcb8e5e" alt="project-desc-p2" width="499">
    </td>
  </tr>
</table>

### What is Automatic Speech Recognition?
<div align="justified">
  <p>
    Automatic Speech Recognition (ASR) converts spoken language into text, widely utilized in applications like virtual assistants and transcription services. Traditional ASR relied on Hidden Markov Models (HMMs) and Gaussian Mixture Models (GMMs), but the advent of deep learning revolutionized the field. Neural networks, particularly Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs), improved the recognition of complex speech patterns. Recent advancements include Transformer models, which leverage self-attention mechanisms for handling large contexts and dependencies, exemplified by state-of-the-art systems like OpenAI's Whisper and Facebook AI's Wav2Vec 2.0.
  <div align="center">
    <img src="https://github.com/user-attachments/assets/a83ce077-2a5a-4bac-a3a0-793bd47a1a38" alt="Example ASR Model Wav2Letter" width="800">
  </div>
Despite these advancements, ASR models often perform poorly on accented speech due to biases in training data, which predominantly features standard accents. Accented speech introduces acoustic variability and phonetic differences that are not well-represented in the models' training sets. These models also face limitations in language processing for accented variations, leading to inaccuracies. Additionally, many ASR systems lack sufficient fine-tuning on accented datasets, further exacerbating performance issues.
  <p></p>
  Addressing these challenges requires augmenting training datasets with diverse accented speech, employing transfer learning to fine-tune models on such data, and using domain adaptation techniques to adjust models for specific accents. Phonetic adaptation and user-specific learning can also enhance performance by accounting for individual speech patterns. By implementing these strategies, ASR technology can become more inclusive and effective across different linguistic backgrounds, improving its applicability in real-world scenarios.
  </p>
</div>

### Accented English Datasets
<div align="justified">
  <p>
    Accented English refers to variations in pronunciation, intonation, and stress patterns influenced by a speaker's native language or regional background. These accents can significantly differ from standard or neutral English, posing challenges for ASR systems. African-American Vernacular English (AAVE) is a prominent example, characterized by unique grammatical, phonological, and syntactical features. AAVE's distinctiveness stems from historical and cultural influences, making it a vital dialect for linguistic study and representation in ASR systems. Similarly, Indic Accented English, spoken by individuals from the Indian subcontinent, incorporates phonetic and intonational patterns influenced by native Indic languages, leading to distinct variations in English pronunciation.
    <p>
    </p>
The Corpus of Regional African American Language (CORAAL) is a comprehensive dataset capturing the linguistic features of AAVE across different regions and contexts. It provides valuable data for studying the intricacies of AAVE and improving ASR systems' ability to recognize and process this dialect accurately.On the other hand, the Svarah dataset focuses on Indic Accented English, offering a rich collection of speech data from Indian English speakers. Developed by AI4Bharat, Svarah aims to enhance ASR models' performance on Indic English, addressing the specific challenges posed by this accent. Both datasets are instrumental in advancing the inclusivity and accuracy of ASR technology across diverse English accents.

#### *Our goal is to adapt ASR models to capture AAVE and/or IAAE speech and its syntactic features for accurate transcription tasks.*

  <p>
  </p>
  <table align="center" style="background-color: white; border-collapse: collapse;">
  <tr>
    <td align="center" style="padding: 10px; border: none;">
      <img width="496" alt="Screenshot 2024-08-01 at 9 29 27 AM" src="https://github.com/user-attachments/assets/2b2e9f58-a477-4442-b5f8-443c8ef7d8de">
    </td>
    <td align="center" style="padding: 10px; border: none;">
      <img width="382" alt="Screenshot 2024-08-01 at 9 27 31 AM" src="https://github.com/user-attachments/assets/b8779fd2-b208-4540-ac29-ceb02f84cfa7">
    </td>
  </tr>
</table>
  <p>
  </p>
  </p>
</div>

### Fine-tuning Workflow
<div align="justified">
    <div align="center">
      <img width="900" alt="Screenshot 2024-08-01 at 9 34 15 AM" src="https://github.com/user-attachments/assets/9c08969e-efa7-41df-a496-3acf8e014968">
  </div>
    To improve the transcription of accented speech, particularly African-American Vernacular English (AAVE) and Indic Accented English (IAE), we aim to fine-tune the Whisper model by adjusting key hyperparameters. This fine-tuning involves optimizing batch size, learning rate, scheduler, and the number of epochs to enhance the model's performance on accented speech data. By carefully modifying these parameters, we can better adapt the model to the phonetic and intonational characteristics of AAVE and IAE, thereby increasing its transcription accuracy.
  <p>
  </p>
  For preprocessing, we undertake two critical steps. First, we segment the audio to fit the Whisper model's maximum input length of 30 seconds, utilizing the start and end times from our transcript data. This segmentation ensures that the model processes manageable chunks of audio, preventing overflows and maintaining context. Second, we apply zero-padding to both audio and text tensors, ensuring uniformity in tensor size across batches. This padding aligns the data for efficient processing, enabling the model to handle variable-length inputs consistently. These preprocessing steps, combined with fine-tuning the model's hyperparameters, aim to significantly enhance ASR performance for AAVE and IAE.
  <p>
  </p>
  <table align="center" style="background-color: white; border-collapse: collapse;">
  <tr>
    <td align="center" style="padding: 10px; border: none;">
      <img width="499" alt="Screenshot 2024-08-01 at 9 40 06 AM" src="https://github.com/user-attachments/assets/25f202c5-a8ea-488c-82ee-759836e6a439">
    </td>
    <td align="center" style="padding: 10px; border: none;">
    <img width="499" alt="Screenshot 2024-08-01 at 9 40 06 AM" src="https://github.com/user-attachments/assets/daabe736-fab0-4fbd-bba7-a9a43e2ada5f">
    </td>
  </tr>
</table>
  </p>
</div>

### Preliminary Results
<div align="center">

| Model Type                     | Sample Quantity | Loss       | Word Error Rate |
|--------------------------------|------------------|------------|------------------|
| **Pre-trained IAE-tested**      | 680,000 hrs      | 4.3300     | 0.2786           |
| **Pre-trained AAVE-tested**     | 680,000 hrs      | *0.0192    | *0.4234          |
| **IAE-trained**                 | 9.6 hrs          | 4.4142     | 0.2746           |
| **AAVE-trained**                | 8.620 hrs        | *0.2746    | *0.3504          |

**Inference ran on a small subset due to resource-limitations and timeline constraints.*

</div>

<div align="justified">  
  The preliminary results of our study reveal promising trends in the fine-tuning process of the Whisper model for accented speech recognition. The Training Loss per Epoch graph demonstrates a steady decline in training loss as the number of epochs increases, indicating effective learning and improved performance on the training data. This trend is mirrored in the Training Word Error Rate (WER) per Epoch graph, which shows a consistent decrease in WER, reflecting enhanced accuracy in the model's predictions during training.
  <div align="center">
    <img width="900" alt="Screenshot 2024-08-01 at 9 34 15 AM" src="https://github.com/user-attachments/assets/d61ffef5-c779-43c0-acf3-23c46be1c99f">
  </div>
  <p>
  </p>
  The validation metrics further support these positive outcomes. The Validation Loss per Epoch graph exhibits a general downward trend, although with some fluctuations, suggesting that the model is performing well on unseen validation data. These variations indicate that additional tuning may be required to stabilize the learning process. Similarly, the Validation Word Error Rate (WER) per Epoch graph shows a declining trend with some variability, highlighting areas where model adjustments or more data could improve consistency and overall performance.
  <div align="center">
    <img width="900" alt="Screenshot 2024-08-01 at 9 34 15 AM" src="https://github.com/user-attachments/assets/bc567be8-ac2b-4171-8ae7-093526c57912">
  </div>
  <p>
  </p>
  </p>
</div>

### Conclusion and Discussion
1. The models demonstrate no significant performance or accuracy difference between pre-trained and IAAE-trained model. However, significantly different sizes of data were used in each model.
1. Results suggest potential of diverse English accents in training datasets can enhance ASR technology, making it more inclusive and accurate.
1. Use of IAAE dataset in our ASR model suggest potential for ASR systems to be tailored to better serve underrepresented speech patterns, thereby addressing equity issues in technology.
1. Detailed error analysis revealed common transcription errors related to specific phonetic nuances of IAAE, guiding future model enhancements

### Future Work
1. Perform fine-tuning on AAVE Speech and compare performance against SOTA AAVE-trained models.
1. Extend research on other accented English. (Latin American, European, etc.)
1. Explore optimization techniques for model performance on resource-limited environments.
1. Develop mechanisms to apply user-feedback for continuous fine-tuning of evolving speech patterns.

### Usage Instructions
1. `Pending`

### Poster Presentation
<div align="center">

  ***Poster Presentation was showcased on August 2nd, 2024 at the University of Southern California Viterbi School of Engineering Research Symposium.***
  
  <img width="3401" alt="image" src="https://github.com/user-attachments/assets/1375b666-d96a-4e7b-bcba-f8b3ea0d0f6c">
</div>

### Oral Presentation
<div align="center">

  ***Oral Presentation was showcased on August 1st, 2024 at the NACME Apple Artificial Intelligence-Machine Learning Intensive Bootcamp 2024 Final Project Showcase.***
  
  <img width="1470" alt="Screenshot 2024-08-01 at 12 22 48 PM 1" src="https://github.com/user-attachments/assets/72437b89-0ca9-478e-a979-3e46a01eb8df">
</div>

### References
1. J. Baugh, *Beyond Ebonics: Linguistic Pride and Racial Prejudice*. Oxford University Press, 2000.
2. W. Labov, *Language in the Inner City: Studies in the Black English Vernacular*. University of Pennsylvania Press, 1972.
3. J. R. Rickford, *African American Vernacular English: Features, Evolution, Educational Implications*. Wiley-Blackwell, 1999.
4. D. S. Ureña and K. Schindler, "CORAAL: The Corpus of Regional African American Language," University of Oregon, 2022. [Online]. Available: https://oraal.uoregon.edu/coraal. [Accessed: 01-Aug-2024].
5. AI4Bharat, "Svarah: A Large Scale Indic Speech Recognition Dataset," 2022. [Online]. Available: https://github.com/AI4Bharat/Svarah. [Accessed: 01-Aug-2024].
6. G. Hinton, O. Vinyals, and J. Dean, "Distilling the Knowledge in a Neural Network," *arXiv preprint arXiv:1503.02531*, 2015.
7. A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, "Attention is all you need," in *Advances in Neural Information Processing Systems*, 2017, pp. 5998-6008.
8. A. Radford, K. Narasimhan, T. Salimans, and I. Sutskever, "Improving Language Understanding by Generative Pre-Training," *OpenAI*, 2018.
9. A. Baevski, Y. Zhou, A. Mohamed, and M. Auli, "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations," in *Advances in Neural Information Processing Systems*, 2020.
10. Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer, and V. Stoyanov, "RoBERTa: A Robustly Optimized BERT Pretraining Approach," *arXiv preprint arXiv:1907.11692*, 2019.
11. J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," *arXiv preprint arXiv:1810.04805*, 2018.
12. S. Karita, N. Chen, T. Hayashi, T. Hori, H. Inaguma, Z. Jiang, M. Someki, N. S. Moritz, P. W. Chan, and S. Watanabe, "A Comparative Study on Transformer vs RNN in Speech Applications," *arXiv preprint arXiv:1909.06317*, 2019.
13. A. Gulati, J. Qin, C.-C. Chiu, N. Parmar, Y. Zhang, J. Yu, W. Han, S. Wang, Z. Zhang, Y. Wu, and R. Pang, "Conformer: Convolution-augmented Transformer for Speech Recognition," *arXiv preprint arXiv:2005.08100*, 2020.
14. S. Wang, Y. Wu, X. Xu, Y. Xie, Y. Zhang, and H. Meng, "Fine-tuning Bidirectional Encoder Representations from Transformers (BERT) for Context-Dependent End-to-End ASR," *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, vol. 29, pp. 2938-2950, 2021.
15. Kaldi ASR Toolkit, "Kaldi: A toolkit for speech recognition," 2011. [Online]. Available: http://kaldi-asr.org. [Accessed: 01-Aug-2024].
16. A. Graves, A.-R. Mohamed, and G. Hinton, "Speech recognition with deep recurrent neural networks," in *2013 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2013, pp. 6645-6649.
17. A. Hannun, C. Case, J. Casper, B. Catanzaro, G. Diamos, E. Elsen, R. Prenger, S. Satheesh, S. Sengupta, A. Coates, and A. Ng, "Deep Speech: Scaling up end-to-end speech recognition," *arXiv preprint arXiv:1412.5567*, 2014.
18. W. Chan, N. Jaitly, Q. V. Le, and O. Vinyals, "Listen, attend and spell: A neural network for large vocabulary conversational speech recognition," in *2016 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2016, pp. 4960-4964.
19. S. Niu, B. Tang, Z. Cheng, X. Chen, J. Zeng, Y. Huangfu, and H. Meng, "Improving Speech Recognition Systems for Accented English Speech with Multi-Accent Data Augmentation," *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, vol. 28, pp. 1679-1692, 2020.
20. R. Maas and D. Hovy, "Adapting Deep Learning Models for Accented Speech Recognition," in *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 34, no. 05, pp. 7969-7976, 2020.
