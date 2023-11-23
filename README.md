![header](https://capsule-render.vercel.app/api?type=shark&color=auto&height=200&section=header&text=CLIP%20study&fontSize=60&&animation=fadeIn&fontAlignY=30)

## Paper Review on CLIP



  
### Prompt + CLIP
<details>
<summary>R-Tuning: Regularized Prompt Tuning in Open-Set Scenarios</summary> <br/> 
  
* **Problem**
  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: OOD performance drops significantly with the increase of the in-distribution classes



* **Method**


     &nbsp;&nbsp;&nbsp;1) Extend the range of words forming texts (R-Tuning)
    <img width="600" alt="스크린샷 2023-11-08 오후 6 54 33" src="https://github.com/mjoo-lee/CLIP_study/assets/110808006/0f9338f3-5df8-4815-bf1a-e331dd83c2fd">

     &nbsp;&nbsp;&nbsp;2) Combinatorial Tuning and Testing (CTT)

    <img width="600" alt="스크린샷 2023-11-08 오후 6 57 55" src="https://github.com/mjoo-lee/CLIP_study/assets/110808006/a1ff0ba8-90bb-43c9-9c89-1490a20d7424"><br/> 



* **Experiment Setting**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: Unknown Detection + Detailed Open-set Recognition 

</details>


<details>
<summary>Improving Zero-Shot Generalization for CLIP with Synthesized Prompts(SHIP) (ICCV '23) </summary><br/>

  
* **Problem**
  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: Develop a fine-tuning approach that can effectively recognize both categories with/without available data


* **Method**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Model learns the mapping from visual features to token embedding space via the VAE process
<img width="607" alt="스크린샷 2023-11-08 오후 8 40 26" src="https://github.com/mjoo-lee/CLIP_study/assets/110808006/183a1116-cd72-4b03-9207-84e380d7af38">



* **Experiment Settings**

  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;bse-to-new + cross-dataset transfer(trained on ImageNet, evalueated on target datasets) </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ generalized zero-shot classification(trained on seen class, evaluated on mixture of seen/unseen class test dataset)

</details>

<details>
<summary>Read-only Prompt Optimization for Vision-Language Few-shot Learning</summary><br/>

  
* **Problem**
  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: Prompt tuning methods shift internal representation through attention mechanism, which negatively impact the robustness and</br> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;generalization of the model in data-deficient(=few-shot) settings.


* **Method**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Read-only Prompt Optimization(RPO)</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Prompts concatenated to the input of the visual and text encoders are processed with masked attention to avoid the impact on the &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;internal representation of CLIP.


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img width="635" alt="스크린샷 2023-11-08 오후 8 50 58" src="https://github.com/mjoo-lee/CLIP_study/assets/110808006/4f27d9c9-1f39-49be-8cb3-3f520c98e94c">&nbsp;&nbsp;&nbsp;<img width="323" alt="스크린샷 2023-11-08 오후 8 52 40" src="https://github.com/mjoo-lee/CLIP_study/assets/110808006/af3e358a-1d07-463e-aea7-1003ab87088a">


* **Experiment Settings**

  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Base-to-new generalization(16-shot) + Domain generalization

</details>


<details>
<summary>TBD</summary><br/>

  
* **Problem**
  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: Ability to infer samples that are not in its training data distribution is still weak.


* **Method**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;



* **Experiment Settings**
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

</details>



-------
------

### Continual learning + CLIP
<details>
<summary>Don't stop learning: Towards Continual Learning for the CLIP Model</summary><br/>

* **Problem**
  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: Decreased zero-shot performance after fine-tuning


* **Method**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Learning without Forgetting via Replayed Vocabularies (VR-LwF)
    <img width="542" alt="스크린샷 2023-11-08 오후 7 20 13" src="https://github.com/mjoo-lee/CLIP_study/assets/110808006/4dd3f9af-f8fe-4538-bd0c-1d38fc5386dd">



* **Experiment Settings**
  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: UT-Acc (updated task) + ZS-Acc (zero-shot) + Retrieval + Bwt(Backward transfer)
</details>


<details>
<summary>S-Prompts Learning with Pre-trained Transformers: An Occam’s Razor for Domain Incremental Learning (NeurIPS '22)</summary><br/>
  
* **Problem**
  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: Prompting methods(DyTox, L2P) aim at learning task/domain-specific prompts dependently across domains, leading to mix up in &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;the subspaces of old/new knowledge


* **Method**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Learned expert knowledge for each domain is finally gathered in a pool
 <img width="500" alt="스크린샷 2023-11-08 오후 7 50 52" src="https://github.com/mjoo-lee/CLIP_study/assets/110808006/d842034c-a915-45a4-b5d8-60fa82d5a389">  <img width="500" alt="스크린샷 2023-11-08 오후 7 51 32" src="https://github.com/mjoo-lee/CLIP_study/assets/110808006/9aafce57-a36d-4b25-b33b-5bbf5d0baf9a">

 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1) K-Means to store the centroids for each domain during training</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2) K-NN to search for the nearest centroid of the given test image feature to identify its domain during inference.</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3) Both K-Means and K-NN are performed on the feature space of the fixed pre-trained transformer without using any prompt



* **Experiment Settings**


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;DIL (Forward Classification Accuracy)

</details>





<details>
<summary>Continual Vision-Language Representation Learning with Off-Diagonal Information</summary><br/>

  
* **Problem**
  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: Ability to infer samples that are not in its training data distribution is still weak.


* **Method**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;✔️ Analyze the changes in the model’s representation space from a spatial geometry perspective during continual CLIP training 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1) Intra-modal rotation : same sample's vision representation vector from different training phases' vision encoder</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(Rotation-Angle Matrix(RAM))


   &nbsp;&nbsp;&nbsp;&nbsp;<img width="328" alt="스크린샷 2023-11-08 오후 8 10 44" src="https://github.com/mjoo-lee/CLIP_study/assets/110808006/6fcd05ba-4e0b-46e5-be9a-4c59ae8bd9c1">

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2) Inter-modal rotation : space rotations of the two encoders from different training phases


   &nbsp;&nbsp;&nbsp;&nbsp;<img width="323" alt="스크린샷 2023-11-08 오후 8 20 22" src="https://github.com/mjoo-lee/CLIP_study/assets/110808006/f52e654e-ad71-4b0c-82ac-71e887bc16ff"></br>



  &nbsp;&nbsp;&nbsp;&nbsp;<img width="590" alt="스크린샷 2023-11-08 오후 8 20 45" src="https://github.com/mjoo-lee/CLIP_study/assets/110808006/6bbaa5ee-61a5-444a-a06d-3f21f70a10d2">



* **Experiment Settings**


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Train CLIP on sub-datasets sequentially
</details>


<details>
<summary>CODA-Prompt: COntinual Decomposed Attention-based Prompting for Rehearsal-Free Continual Learning (CVPR '23) </summary><br/>

  
* **Problem**
  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: Prompt-based approaches(L2P, DualPrompt) reduce forgetting by sacrificing new task accuracy. </br> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Also, Expanding prompt size does not increase the plasticity.


* **Method**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img width="258" alt="스크린샷 2023-11-15 오후 2 47 50" src="https://github.com/mjoo-lee/CLIP_study/assets/110808006/9b65888c-cdf6-44c7-b120-911a499cdbe5">
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <img width="489" alt="스크린샷 2023-11-15 오후 2 48 14" src="https://github.com/mjoo-lee/CLIP_study/assets/110808006/4bd97cda-6a74-4043-9db6-565af8ba23c3"></br> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ✔️ Replace learnable prompt parameter p with a weighted summation over the prompt components </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ✔️ Loss : <img width="244" alt="스크린샷 2023-11-15 오후 2 50 57" src="https://github.com/mjoo-lee/CLIP_study/assets/110808006/dfb0854f-cb62-47d2-a3bf-780e5c2bdb53"></br></br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; :star2: Interesting point : orthogonality of accumulated prompts help continual learning



* **Experiment Settings**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Class incremental learning + dual-shift (random selection of classes & domains during each task)

</details>



<details>
<summary>AttriCLIP: A Non-Incremental Learner for Incremental Knowledge Learning ('23 CVPR) </summary><br/>

  
* **Problem**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: There exist limitations of conventional continual learning </br> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1) Learns sequentially arrived tasks or classes with a shared model (**overwriting** → catastrophic forgetting) </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2) Classifier needs to be expanded to recognize novel classes </br></br> 



* **Method**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img width="400" alt="스크린샷 2023-11-23 오후 2 23 06" src="https://github.com/mjoo-lee/CLIP_study/assets/110808006/71c8bd7a-5c31-4ea2-a5e0-06aebbca4fc0"> </br> 
------------------------
<img width="600" alt="스크린샷 2023-11-23 오후 2 22 51" src="https://github.com/mjoo-lee/CLIP_study/assets/110808006/240b3d2d-3793-476b-9789-48c52c398be3">




* **Experiment Settings**
  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;✔️ Datasets </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• CIFAR100, ImageNet100 : split into 10 tasks with 10 classes in each task </br>  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;✔️ Training details</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• 10 epochs on each incremental task for all dataset </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• Prompt length M= 12</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• Number of attributes in the bank N= 10</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• Number of selected attributed C= 3</br></br>




* **Feedbacks**
  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; How did it prevent catastrophic forgetting? 
</details>


<details>
<summary>TBD</summary><br/>

  
* **Problem**
  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: 


* **Method**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;



* **Experiment Settings**
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

</details>



<details>
<summary>TBD</summary><br/>

  
* **Problem**
  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;:


* **Method**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;



* **Experiment Settings**
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

</details>


<details>
<summary>TBD</summary><br/>

  
* **Problem**
  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: 


* **Method**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;



* **Experiment Settings**
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

</details>

<details>
<summary>TBD</summary><br/>

  
* **Problem**
  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: A


* **Method**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;



* **Experiment Settings**
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

</details>






-------
------
### Etc.
<details>
<summary>Preventing Zero-shot Transfer Degradation in Continual Learning of Vision-Language Models(ICCV '23)</summary><br/>

  
* **Problem**
  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: Increased continual learning performance at the cost of sacrificing zero-shot performance


* **Method**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1) Distillation in Feature Space </br> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img width="498" alt="스크린샷 2023-11-23 오후 2 10 14" src="https://github.com/mjoo-lee/CLIP_study/assets/110808006/417f58f0-ea3c-40f1-9ae3-61137f7434a9">  </br> </br> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;✔️ Data source - Reference dataset, general image(ImageNet)  </br> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;✔️ Teacher model - Pretrained CLIP  </br> </br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2) Weight Ensemble in Parameter Space </br> </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img width="246" alt="스크린샷 2023-11-23 오후 2 11 54" src="https://github.com/mjoo-lee/CLIP_study/assets/110808006/78a80e91-06f6-4e7d-b1f4-12364eac9d5e"> </br>


* **Experiment Settings**
  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Multi-domain Task Incremental Learning (MTIL) </br></br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;✔️ Consists of 11 tasks, total number of 1,201 classes </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img width="302" alt="스크린샷 2023-11-23 오후 2 13 46" src="https://github.com/mjoo-lee/CLIP_study/assets/110808006/a11d9471-c509-4938-af5e-6475ada68edd">&nbsp;&nbsp;<img width="400" alt="스크린샷 2023-11-23 오후 2 17 48" src="https://github.com/mjoo-lee/CLIP_study/assets/110808006/cd6f0ad1-b2c1-451d-bd04-189fbf189343">


</details>

<details>
<summary>TBD</summary><br/>

  
* **Problem**
  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: 


* **Method**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;



* **Experiment Settings**
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

</details>

<details>
<summary>TBD</summary><br/>

  
* **Problem**
  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: 


* **Method**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;



* **Experiment Settings**
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

</details>
![Uploading image.png…]()
