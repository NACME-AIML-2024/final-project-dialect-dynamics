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
  <p>
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

### For more information...
Feel free to contact us at any time or reach out via this repo


