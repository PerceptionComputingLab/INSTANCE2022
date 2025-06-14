![](https://github.com/PerceptionComputingLab/INSTANCE2022/blob/main/images/Logo_%E9%A1%B5%E9%9D%A2_1_RCTKLKW.png)

## 🚀Important News

🎉**NEWS [2022/07/19]:** Extended the evaluation phase to August 7th. You can submit your validation results on the submission page until August 7th. And you can submit your docker and paper from July 20th to August 14th.

🎉**NEWS [2022/07/15]:** Validation dataset release. We have sent the data download link via email. If you didn't receive the email, please let us know as soon as possible!

🎉**NEWS [2022/04/06]:** Training dataset release.

🎉**NEWS [2022/03/28]:** INSTANCE challenge 2022  is now open for [registration](https://instance.grand-challenge.org)! Remember to send the [signed document](https://github.com/PerceptionComputingLab/INSTANCE2022/blob/main/Agreements/instance2022_agreements.pdf) to INSTANCE2022@outlook.com for participation.

## 🏊‍About

Intracranial hemorrhage (ICH) is a common stroke type and has the highest mortality rate among all stroke types[1]. Early and accurate diagnosis of the ICH is critical for saving patients' lives. In regular clinical practice, Non-Contrast Computed Tomography (NCCT) is the most widely used modality for diagnosing ICH due to its fast acquisition and availability in most emergency departments [2]. In clinical diagnosis procedures, accurately estimating the volume of intracranial hemorrhage is significant for predicting hematoma progression and early mortality [3]. The hematoma volume can be estimated by manually delineating the ICH region by radiologists, which is time-consuming  and suffers from inter-rater variability . The ABC/2 method [4] is widely adopted in clinical practice to estimate hemorrhage volume for its ease of use. However, the ABC/2 method shows significant volume estimation error, especially for those hemorrhages with irregular shapes. Hence, it is necessary to establish a fully-automated segmentation method, which allows accurate and rapid volume quantification of the intracranial hemorrhage. However, it is still challenging to accurately segment the ICH for automatic methods because ICH exhibits large variations in shapes and locations, and has blurred boundarie.

Therefore, it is significant to propose a challenge to advance the novel automatic segmentation methods for accurate intracranial hemorrhage segmentations on NCCT images. However, there are no such challenges available right now. Thus, we intend to host the first Intracranial Hemorrhage Segmentation Challenge on Noncontrast head CT (Named INSTANCE 2022), which will be served as a solid benchmark for Intracranial hemorrhage Segmentation tasks.

Specifically, we have collected 200 3D volumes with refined labeling from 10 experienced radiologists, 100 for the training dataset, 70 for the closed testing dataset, and 30 for the opened validated dataset. DSC, HD, RVD are adopted as evaluation metrics for segmentation. This challenge will also promote intracranial hemorrhage treatment, interactions between researchers, and interdisciplinary communication.

[1] C. J. van Asch, M. J. Luitse, G. J. Rinkel, I. van der Tweel, A. Algra, and C. J. Klijn, “Incidence, case fatality, and functional outcome of intracerebral haemorrhage over time, according to age, sex, and ethnic origin: A systematic review and meta-analysis,” Lancet. Neurol., vol. 9, no. 2, pp. 167–176, Feb. 2010.  
[2] J. N. Goldstein and A. J. Gilson, “Critical care management of acute intracerebral hemorrhage,” Curr. Treat. Option. Neurol., vol. 13, no. 2, pp. 204–216, Jan. 2011.  
[3] J. P. Broderick, T. G. Brott, J. E. Duldner, T. Tomsick, and G. Huster, “Volume of intracerebral hemorrhage. A powerful and easy-to-use predictor of 30-day mortality,” Stroke, vol. 24, no. 7, pp. 987–993, Jan. 1993.  
[4] R. U. Kothari et al., “The ABCs of measuring intracerebral hemorrhage volumes,” Stroke, vol. 27, no. 8, pp. 1304–1305, Aug. 1996.

## 🏹Task
Participants are required to segment Intracranial Hemorrhage region in Non-Contrast head CT (NCCT).

## 🤗Schedule
- Registration: March 28 (11:59PM GMT), 2022
- Training dataset release: April 6 (11:59PM GMT), 2022
- Validation dataset release, open validation leaderboard submission: July 15 (11:59PM GMT), 2022
- Deadline for the validation leaderboard submission: Aug 7 (11:59PM GMT), 2022
- Opening docker and short paper submission for testing dataset: July 20 (11:59PM GMT), 2022
- Deadline for docker and short paper submission: Aug 14 (11:59PM GMT), 2022
- Winner and invitation speakers: September 18 (11:59PM GMT), 2022

![](https://github.com/PerceptionComputingLab/INSTANCE2022/blob/main/images/image_rXCO79b.png)

## 🔭Registration
Individuals or team members interested in participating in this challenge should carefully study the [challenge rules](https://instance.grand-challenge.org/Participation/) and then follow the instructions to join challenge.

*Please note that by participating in this challenge you are agreeing to all its rules and policies.*

## 🏆Award
1. Successful participation awards, which are electronic certificates, will be awarded to all teams that obtain valid test scores in the challenge leaderboard and complete technical paper submissions reviewed by the organizing committee.
2. The top-1 team will receive 500 dollars or electronic products with similar prices. The exquisite certificates will be awarded to all members of the Top-1 team.
3. The team that wins the second place will receive 300 dollars or electronic products with similar prices. The exquisite certificates will be awarded to all members of the Top-2 team.
4. The team that wins the third place will receive 200 dollars or electronic products with similar prices. The exquisite certificates will be awarded to all members of the Top-3 team.
5. The team achieving the first place in the single index (such as Dice, HD, or RVD) will be awarded to all members with electronic certificates.

*Noted that we will not award those who refuse to make presentations during MICCAI. However, those participants don't have to show in person, they can send a representative to make the presentations.*

## 🌱 Citation
If using our dataset, you must cite the following paper:

[1] Li, X., Luo, G., Wang, K., Wang, H., Li, S., Liu, J., Liang, X., Jiang, J., Song, Z., Zheng, C., Chi, H., Xu, M., He, Y., Ma, X., Guo, J., Liu, Y., Li, C., Chen, Z., Siddiquee, M.M., Myronenko, A., Sanner, A.P., Mukhopadhyay, A., Othman, A.E., Zhao, X., Liu, W., Zhang, J., Ma, X., Liu, Q., MacIntosh, B.J., Liang, W., Mazher, M., Qayyum, A., Abramova, V., & Llad'o, X. (2023). The state-of-the-art 3D anisotropic intracranial hemorrhage segmentation on non-contrast head CT: The INSTANCE challenge. ArXiv, abs/2301.03281. [[Arxiv]](https://arxiv.org/abs/2301.03281)  
[2] X. Li, G. Luo, W. Wang, K. Wang, Y. Gao and S. Li, "Hematoma Expansion Context Guided Intracranial Hemorrhage Segmentation and Uncertainty Estimation," in IEEE Journal of Biomedical and Health Informatics, vol. 26, no. 3, pp. 1140-1151, March 2022, doi: 10.1109/JBHI.2021.3103850. [[Paper]](https://ieeexplore.ieee.org/abstract/document/9511297)
