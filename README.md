# Speckle-Noise-removal
This is implementation of https://rdcu.be/dmaMY

Wavelet and Total Variation Based Method Using Adaptive Regularization for Speckle Noise Reduction in Ultrasound Images
Nishtha Rawat, Manminder Singh & Birmohan Singh 
Wireless Personal Communications volume 106, pages1547–1572 (2019)Cite this article


Abstract
Ultrasound (US) images are useful in medical diagnosis. US is preferred over other medical diagnosis technique because it is non-invasive in nature and has low cost. The presence of speckle noise in US images degrades its usefulness. A method that reduces the speckle noise in US images can help in correct diagnosis. This method also should preserve the important structural information in US images while removing the speckle noise. In this paper, a method for removing speckle noise using a combination of wavelet, total variation (TV) and morphological operations has been proposed. The proposed method achieves denoising by combining the advantages of the wavelet, TV and morphological operations along with the utilization of adaptive regularization parameter which controls the amount of smoothing during denoising. The work in this paper has the capability of reducing speckle noise while preserving the structural information in the denoised image. The proposed method demonstrates strong denoising for synthetic and real ultrasound images, which is also supported by the results of various quantitative measures and visual inspection.

Access provided by University of Missouri - St Louis Libraries

1 Introduction
There are many methods which are used for medical diagnosis out of which Ultrasound (US) imaging is the most widely used method which has the benefit over other medical diagnostic methods, as the acquisition of these images are easy and less costly. While obtaining an US image, it gets corrupted with speckle noise [1]. Generation of speckle noise is due to the interference of beams reflected towards the transducer, which causes a granular artifact to appear in the image. Due to the speckle noise, the important structural information contained in the US image gets corrupted and thus the examiner may not be able to interpret the image correctly. Speckle noise also contributes to low image resolution and contrast which further leads to poor quality of US images. Thus, removing the speckle noise from US images is an important step [2].

The US are sound waves above 20 kHz, which exceed the audible frequency range. For medical US imaging, the range is between 2 and 18 MHz. The pulse-echo effect is used in US imaging, the steps of which are transmitting, listening and receiving. Initially, the ultrasonic pulses are transmitted into the organ via a transducer. After the pulses hit an organ boundary, these are reflected back from the tissues. These reflected echoes are processed to form an image. But, there are certain echoes that get scattered which then results in an interference pattern called speckle noise [3].

2 Literature Review
Over many years, research on speckle noise reduction has been done on a wide scale. For removal of noise, linear and non-linear filters have been used. Non-linear filters work on the concept of Partial Differential Equations while linear filters work on the concept of averaging. Non-linear filters generally contain methods based on diffusion such as isotropic diffusion, anisotropic diffusion, and total variation (TV). Some linear filters are Median filter, Nonlocal-Means filter and Lee filter. For image filtering, there are also multiscale spatial filtering methods which contain wavelet, curvelet and ridgelet transform. We have divided the literature review into three parts. The first part contains the work done till now for speckle noise removal using wavelets, the second part contains the work done using diffusion and the third part contains papers related to the hybrid of wavelet and diffusion methods.

2.1 Literature Review Wavelet
For speckle noise reduction in US images wavelets have been extensively used. Wavelet uses the concept of decomposing the images and then working on its component for noise removal and then assembling it back into the original image without loss of information. The Efficiency of wavelets in removing speckle noise makes it a good filter. The related work done in the speckle noise removal using wavelets in recent years are mentioned below:

In the year 2001, Achim et al. introduced a multiscale nonlinear homomorphic method for speckle reduction in US images which used log transform and wavelet transform for noise removal along with the use of Bayesian estimator. They compared their method with a median filter, homomorphic Wiener filter, and wavelet shrinkage denoising [4]. In the year 2004, Gupta et al. developed a method that used the wavelet transform with Generalized Gaussian Distribution. They used sub-band modeling and used soft thresholding to threshold the wavelet coefficients. The threshold value was evaluated using scale parameter, they also used data from sub-band of noise-free image and the value of standard deviation of noise for calculation of threshold. Comparison of their method was done with median filter [5]. In the year 2006, Chen and Raheja developed a method that used Generalized Cross Validation thresholding technique and wavelet transforms for removal of speckle noise from US images. They used thresholding which was level dependent and for each sub-band in wavelet domain, the noise energy was automatically estimated [6]. In the year 2009, Sudha et al. developed a method that used thresholding method using wavelets for speckle noise reduction [7]. In the same year, Mateo and Fernández-Caballero introduced hybrid methods formed by combinations of a median filter, Fourier filter, wavelet transform and Homomorphic filter along with different window size for smoothing and suppression of speckle noise in US images and their performance evaluation was done using quantitative evaluation techniques MSE and PSNR [8]. In the year 2011, Sarode and Deshmukh used the concept of Discrete Wavelet Transform (DWT) for removal of speckle noise and calculated threshold for sub-band coefficients [9]. In the same year, Ruikar and Doye computed efficient wavelet coefficient along with the new threshold function for speckle noise reduction [10]. In the year 2012, Andria et al. developed method based on linear filtering with use of a Gaussian filter on wavelet coefficients along with vertical and diagonal details [11]. In the year 2013, Joel and Sivakumar introduced techniques for reducing speckle noise in US images which included details of spatial filtering methods such as Lee and median. Also, multiscale filtering methods like Wavelet transform, curvelet transform and contourlet transform was used [12]. In the year 2015, Yadav et al. provided an analysis of DWT in combination with different wavelet families and developed a method that used combination of median and wiener filter. The combination is further applied with wavelets of type Haar, Symlet to improve image visually [13]. In the year 2016, Zhang et al. computed an optimal threshold for wavelet and then retained wavelet band by using a guided filter, they used PSNR, SSIM for parameter evaluation [14]. In the year 2018, Gai et al. removed speckle noise from US images by developing a method that used Monogenic Wavelet Transform along with Bayesian framework. The components speckle noise and components of noise-free were used as the monogenic coefficients. These coefficients were further used in Laplace mixture distribution and Rayleigh distribution. The use of these distributions along with new Bayesian estimator was done for the speckle noise reduction [15].

2.2 Literature Review for Diffusion
In recent years, many methods have been proposed for diffusion, which has the benefit of reducing speckle noise in the US images along with preserving edge and texture details.

In the year 1990, Perona and Malik introduced diffusion process through scale space technique and developed Perona Malik Anisotropic Diffusion method. This method performed smoothing in US images and preserved the edges [16]. In the year 2002, Yu and Action developed a method that used local statistics of image for reducing noise. This method named as speckle reducing anisotropic diffusion (SRAD) was an edge sensitive diffusion method [17]. In the year 2004, Tauber et al. developed anisotropic diffusion technique by using SRAD. Their method was robust and adaptable to removing speckle noise [18]. In the year 2006, Aja-Fernández and Alberola-López mentioned issues concerning anisotropic diffusion such as estimation of variation coefficients of noise and developed Detail Preserving Anisotropic Diffusion filter [19]. In the year 2007, Krissian et al. used the SRAD filter for creating a matrix anisotropic diffusion filter, which was capable of applying filtering up to various level to the image across the principal curvature directions and its contours [20]. In the year 2011, Liu et al. developed a method that used Tukey error norm function for calculation in Detail Preserving Anisotropic Diffusion filter resulting in a Robust Detail Preserving Anisotropic Diffusion filter [21]. In the year 2014, Toufique et al. developed a method based on the anisotropic diffusion equation of Perona Malik (PM) which was able to determine the diffusion function parameter, scale space adaptively. Intensity variance for each image pixel was computed for adaptation of the method [22]. In the year 2015, Ramos-Llordén developed a filter that handles the over-filtering problem, this filter based on anisotropic diffusion reduce over-filtering by utilizing probabilistic-driven memory. They have also used a tissue-selective mechanism which captures details related to tissue [23]. In the year 2016, Hu and Tang developed a method to reduce speckle noise that utilizes the K-means clustering algorithm to develop a method named as Cluster-Driven Anisotropic Diffusion. It was able to choose homogeneous sample region automatically [24]. In the year 2016, Fredj and Malek developed a filter which maintained the image quality and reduced the processing time named as faster oriented speckle reducing anisotropic diffusion (FOSRAD). Their method used look-ahead decomposition technique for optimizing the processing time [25].

2.3 Literature Review Combination of Wavelet and Total Variation
In recent years, many methods have been proposed using wavelets that have the benefit of preserving the image feature while reducing noise. Also, there have been many methods that use diffusion which can remove speckle noise efficiently. Filtering with use of single step of time discretization in diffusion is regarded as regularization [26]. There has been work done on the combination of both wavelets and diffusion methods.

In the year 2006, Yue et al. developed a method that used wavelet with nonlinear diffusion to reduce speckle noise in US images. They combined multiresolution property of wavelet along with edge enhancement feature of diffusion. Multiscale diffusion process was applied to wavelet coefficients for speckle reduction [27]. In the same year, Wang and Zhou developed a method that used the combination of wavelet and TV for speckle noise reduction. Wavelet coefficients were modified in the high-frequency domain and were selected in order to minimize the TV norm of the images obtained through reconstruction [28]. In the year 2008, Bhoi and Meher developed a method that used wavelet and TV methods for removal of additive white Gaussian noise that appears in images. The decomposed image sub-bands were used to find edges in the image and they then applied inverse wavelet transform to build the image [29]. In the year 2009, Huang et al. introduced a method in which for the first step Newton method was used to find a solution to the nonlinear equation which was convex. For the second step, TV denoising was applied to the image and Chambolle maximization and minimization was used for optimization of the TV function [30]. In the year 2010, Bredies et al. introduced Total Generalized Variation as a regularization term that used derivatives of higher-order. It reduced the staircase effect and regularization was applied on different levels [31]. In the year 2011, Abrahim and Kadah developed a method to remove speckle noise that used the combination of TV and wavelet shrinkage. They decomposed the image into sub-bands using wavelets and removed the low-frequency coefficients using method based on TV [32]. In the same year, Jin and Yang proved the uniqueness of variational problem minimizer and derived the weak solutions to the associated equation [33]. In the year 2013, Xiaorong and Yongjun developed a new method that used multiple wavelet decompositions and the hierarchical threshold for speckle noise reduction. Further to process the decomposed image they used TV method [34]. In the same year, Huang and Yang introduced a method based on model that was convex in nature which used Kullback–Leibler distance to obtain data used for removing speckle noise. This method used variable splitting for faster solving the model along with Bregman iterative method [35]. In the year 2014, Feng et al. introduced a method that combined first and higher order TV functionals and used Total Generalized Variation based methods to eliminate speckle noise. Minimizers of Total Generalized Variation based methods were optimized by the primal-dual algorithm and Newton method [36]. In the year 2016, Elyasi and Pourmina used TV regularization along with DWT and Bayes shrink for speckle reduction [37]. In the year 2017, Wang et al. developed a method that solved the total variation and its high order combination using Alternating Direction Method of Multipliers. Due to objective function being strictly convex the subproblem was then solved by Newton method to find a unique solution [38]. In the year 2018, Mei et al. developed a method to reduce speckle noise that used the data fidelity term combined with the second-order Total Generalized Variation regularization. To solve the constrained optimization problem which was convex, Alternating Direction Method of Multipliers algorithm was used along with Newton method [39].

2.4 Motivation of Work
Speckle noise contributes to low image resolution and contrast, which leads to poor quality of US images. Before analyzing the US image, speckle noise reduction becomes an important pre-processing step. What becomes more important in case of US images is that, the important structural information needs to be preserved. The loss of this information can lead to an incorrect analysis of images or misinterpretation by the examiner. Our main motivation for this work is to reduce speckle noise and focus on maintaining the structural information of the denoised image, along with improving performance evaluation parameters. To achieve this objective, wavelet is used, as wavelet transforms enhance texture and organ surfaces along with denoising. Some amount of denoising can be achieved by wavelets but it is not very efficient if used alone. Thus, TV function has also been used because the amount of denoising achieved by TV function is high and it is capable of removing most of the speckle noise from the US image. Also, it helps in defining shapes and objects in medical imaging. While using TV we may have to deal with its sensitivity to the number of iterations and over-filtering, which may result in the loss of relevant information from US images. To deal with the problem of the number of iterations, two optimization techniques: Newton method and Chambolle method have been used. Over-filtering is removed by making the regularization parameter lambda adaptive which is used in the TV function. The regularization parameter lambda controls the amount of smoothing applied to the image and it is generally set to a fixed value. The fixed value of lambda causes the same amount of smoothing applied to all the images which is not very fruitful. This parameter has been made adaptive so that the amount of smoothing applied is different for all the images used. The value of lambda is obtained from multi wavelet decomposition and using the information from its sub-band HH1 using the median estimator. Thus, it will produce a different value of λ
 according to the amount of noise present in that image. Morphological operations have been used to ensure that the image quality index of US images is maintained, which can be verified by the higher value of UQI parameter. Median filter has been used to maintain the image contrast and the overall image quality. The output is that we get better speckle reduction method which ensures that the structural information of US image is preserved and the denoised image has better parameter value.

The US images that have been taken into consideration in this paper are real kidney US images and carotid artery US images. The kidney is a deeper structure organ in the human body, thus there are chances of more reflections of the ultrasonic pulse while returning to the transducer which results in more speckle contents. Similarly, in case of carotid artery images, there are more edges present. Thus, these two datasets help in achieving the objective during performance evaluation of the proposed method and we are able to judge through visual analysis that how much structural information is preserved.

3 Proposed Method
The proposed method is divided into steps to show the flow of the algorithm:

Step 1 Log transformation.

Speckle noise is multiplicative noise. For converting multiplicative noise to additive noise log transformation is applied. Additive noise is easier to remove than multiplicative noise because noise intensity does not vary with image intensity (Fig. 1).

Fig. 1
figure 1
Block diagram of the proposed method

Full size image
Multiplicative noise is shown as:

M(r,s)=N(r,s)×p(r,s)
(1)
Additive noise is shown as:

M(r,s)=N(r,s)+p(r,s)
(2)
Pixel coordinates of the 2D image are represented by (r,s)
. M(r,s)
 represents the real US image, N(r,s)
 represents the denoised image and p(r,s)
 represents the speckle noise [2].

Log transformation is shown as:

logM=logN+logp
(3)
Step 2 Wavelet transformation.

Wavelet transforms remove speckle noise and retains the detail in the image. There is DWT that corresponds to the decomposition of the image according to its level and inverse DWT for reconstruction of the image after the threshold is applied. The DWT application in an image results in sub-bands (LL, LH, HL and HH). These sub-bands are classified into details and approximation. The details sub-band are LHi, HLi, HHi and the approximation sub-band is LLi, where ‘i’ is the scale, from 1, 2…D, where D represents the total number of decompositions. Wavelet thresholding removes the noise by thresholding the detail sub-bands coefficients and keeping fixed the approximation sub-band coefficients [40].

In the proposed method, DWT has been applied to the image from Eq. (3). Wavelet type ‘Daubechies’ with vanishing moment four and minimum scale three, has been preferred as it is considered to reduce the correlation between the data efficiently. Inverse DWT is applied to reconstruct the denoised image.

Step 3 Total variation regularization.

TV represents the variation in the energy of an image with a function shown as:

E(M,N)=∥M−N∥22+λ|JTV(N)|
(4)
where E is the energy function. ∥⋅∥2
 represents the Euclidean norm, which in general for a term ‘f’ can be represented as ∥f∥2=f21+⋯+f2n−−−−−−−−−−−√
, where (1,2… n) denotes the n-dimensional Euclidean space [30]. M and N are the noisy and denoised image respectively. Lambda λ
 is the regularization parameter that is used to control the smoothing applied in an image. If λ
 is set to a higher value, the denoised image becomes over smoothed and blurred. When λ
 is set to a lower value there is less noise removal from the image. The selection of λ
 value is an important aspect as for every US image the amount of smoothing applied might be different.

In the proposed method, rather than fixing the value of lambda, λ
 is obtained from multi wavelet decomposition of the image M by using the information from its first decomposition level diagonal sub-band HH1. The median estimator method is used to evaluate the noise variance. It does so by using the diagonal detail coefficients to calculate the absolute deviation, where 0.6745 is the median absolute deviation value [41]. Lambda is evaluated as:

λ=[median(|HH1|)0.6745]
(5)
Equation (5) will produce a different value of λ
 according to the amount of noise present in the US image. The amount of smoothing will depend directly on the amount of noise thus removing the effect of blurring from the image. In Eq. (4) JTV(N)
 computes the divergence of N, which is the partial derivative of its respective components [42].

Speckle noise present in the US image increase the energy of the function as shown in Eq. (4), so the denoised Image N is such that it is minimizing the energy of this function. The function obtained from total variation is convex, which indicates that it can converge to only one minimum [43].

In the proposed method, TV function is optimized by using two optimization methods. Energy function used is shown as:

∣∣G(M)V∈Rd∣∣min=12∥M−V∥22+λJ(V)
(6)
For Eq. (6), M is used as a noisy image which is obtained from step 2 and V is the image which is minimizing the overall energy function. V is set equal to the size of the noisy image during the time of calculation of derivative. To solve this Newton method is used. Newton method optimization is a classical optimization technique which converges to a minimum value. The output from Newton method is provided as input for the function shown by Eq. (7).

∣∣G(V)N∈Rd∣∣min=12∥V−N∥22+λJ(N)
(7)
For Eq. (7), V acts as noisy image and this function is solved by Chambolle method for convergence to a minimum value, resulting in the denoised image N.

After the convergence by Chambolle, M is updated to the N in Eq. (6), then the same procedure follows until the stopping criteria is fulfilled.

Newton method optimization is a classical optimization technique which converges to a minimum value. It uses Hessian matrix for first derivative and Jacobian matrix for the second derivative and is simple to use and implement. Chambolle projection optimization is an optimization technique which is used because of its fast convergence and confirms to converge to one minimum if the function used is convex [44]. An advantage of using Chambolle optimization is that we do not need to regularize the TV energy function as done in Euler-Lagrange equations, in which to regularize the energy function we need to calculate this function: ∫|∇V|2+ε2−−−−−−−−−√
, where ε
 is used to avoid numerical instabilities [45].

Step 4 Stopping criteria.

Stopping criteria is fulfilled when outer loop reaches a maximum value that is three (K = 3) or when the condition ∥∥NK+1−NK∥∥2/∥∥NK∥∥2<10−3
 is satisfied. The inner loop for Newton method is fixed at 2 and for Chambolle is fixed at 8.

Step 5 Morphological operations.

Morphological operations are applied to enhance the image quality. The structuring element (SE) represented by ‘s’, plays an important role in defining the object shape. The morphological operation erosion denoted by ⊝
 is used to lighten the edges or boundaries in an image. Morphological operation dilation which is denoted by ⊕
 is used to enlarge the boundaries in an image. Morphological opening denoted by N∘s=(N⊝s)⊕s
 is erosion followed by dilation and morphological closing denoted by N⋅s=(N⊕s)⊝s
 is dilation followed by erosion [46]. In the proposed method, morphological operation closing has been applied on the image obtained from step 3. The structuring element used is a rectangle of size 3×3.

Step 6 Median filter.

The Median filter uses the concept of averaging to remove the noise from images. While removing noise from images it also preserves the edges. It calculates the median of pixels within the window estimated and replaces the center pixel with this median value. This filter is much effective when there are strong spikes like component in the noise pattern [47]. In the proposed method, the median filter has been applied to the image obtained from step 5. The window used is of size 3 × 3.

4 Datasets
The dataset containing synthetic and phantom images is publicly available and has been downloaded from http://field-ii.dk/examples/ftp_files/ [48]. Synthetic and phantom images contain lines, circle or other figures. The dataset containing real kidney US images have been collected from http://www.ultrasoundcases.info/category.aspx?cat=87 [49] and a dataset containing carotid artery images have been collected from http://splab.cz/en/download/databaze/ultrasound [50]. We have used real US images of kidney and carotid artery because these images contain more structural information like of cyst, tumor in case of kidney and edges in case of the carotid artery. Also, speckle noise content is more in US images of the kidney, as the kidney is a deeper structure organ [2]. The proposed method is implemented in MATLAB R2016a.

5 Performance Evaluation Parameters
To measure the performance of the proposed method, the following parameter evaluation techniques have been used where O denotes the original noise-free image or the reference image, N denotes the denoised image and M denotes the noisy image induced with speckle noise. The symbols mean() and var() indicate the mean and variance operations. P and Q denote the size of the image. The pixel coordinates of the image are shown by ‘r’ and ‘s’ where r=1…P
 and s=1…Q
. The technique that requires original noise-free image are known as full-reference metrics, while the metrics using only the denoised image and noisy image are the non-reference metrics.

5.1 Peak Signal to Noise Ratio (PSNR)
It measures the resemblance of the image (O) with the image (N). A larger value of PSNR is preferred [51]. The peak signal to noise ratio equation is represented as:

PSNR=10log10L2MSE
(8)
L is the largest possible value of the intensity in the original image, L is typically 255 or 1.

5.2 Mean Squared Error (MSE)
For an image (O) and image (N), it is used to evaluate the quality change. The lower value of MSE indicates a minimum error [51]. The mean squared error equation is represented as:

MSE=∑Pr=1∑Qs=1[O(r,s)−N(r,s)]2P×Q
(9)
5.3 Root Mean Squared Error (RMSE)
For an image (O) and image (N), it indicates the closeness between the two images. A lower value of RMSE is preferred [2]. The root mean squared error equation is represented as:

RMSE=MSE−−−−−√
(10)
5.4 Universal Quality Index (UQI)
It is defined as a product of three components: loss of correlation, luminance distortion, and contrast distortion, and is used to evaluate the distortion model of the image. The range of values for UQI is [− 1, 1], where 1 is preferred to be the best value indicating the image is of good quality and − 1 indicate the poor image quality [52]. The Universal Quality Index equation is represented as:

UQI=σpqσpσq×2O¯N¯(O¯)2+(N¯)2×2σpσqσ2p+σ2q
(11)
σp,σq
 are standard deviation between the original and denoised image pixel.

5.5 Signal to Noise Ratio (SNR)
The amount of noise content present in the image is measured by SNR. A larger value of SNR indicates that image (N) is less noisy [51]. The signal to noise ratio equation is represented as:

SNR=N(r,s)MSE−−−−−√
(12)
5.6 Mean Absolute Error (MAE)
Small value of MAE is preferred, which indicate that the image is of good quality [53]. The mean absolute error equation is represented as:

MAE=∑Pr=1∑Qs=1|O(r,s)−N(r,s)|P×Q
(13)
5.7 Feature Similarity Index (FSIM)
It is based on advance local image quality measurement and saliency-based weighting. The local image quality is measured by the Phase Congruency (PC) and the Gradient Magnitude (GM) features. A larger value of FSIM indicates good quality image [54]. The Feature Similarity Index equation is represented as

FSIM=∑χ∈Ω[YGM(χ)×YPC(χ)]×PCm(χ)∑χ∈ΩPCm(χ)
(14)
where Ω means the whole image spatial domain. YGM(χ),YPC(χ)
 are GM similarity measure and PC similarity measure respectively.

5.8 Speckle Suppression Index (SSI)
It is a non-reference metric that indicate the speckle content remaining in the image. SSI should be calculated only for homogeneous regions of the image. The lower value of SSI indicates good image and its value should be ideally zero [55]. The Speckle Suppression Index equation is represented as:

SSI=[var(N)−−−−−−√mean(N)]×[mean(M)var(M)−−−−−−−√]
(15)
5.9 Mean Preservation Speckle Suppression Index (MPSSI)
It is also a non-reference metric. This is used because sometimes SSI may not be reliable when mean value is overestimated by the filter. The lower value of MPSSI indicates that the filter has performed better in preserving mean and reducing noise [56]. The Mean Preservation Speckle Suppression Index equation is represented as:

MPSSI=∣∣∣1−mean(N)mean(M)∣∣∣×var(M)−−−−−−−√var(N)−−−−−−√
(16)
5.10 Speckle Suppression and Mean Preservation (SMPI)
To eliminate the effect of overestimation of mean in the denoised image by SSI, SMPI is used which gives an accurate measure. The lower value of SMPI indicates that speckle noise is reduced. It is a non-reference metric [55]. The speckle suppression and Mean Preservation equation is represented as:

SMPI=(Q+|mean(M)−mean(N)|)×var(N)−−−−−−√var(M)−−−−−−−√
(17)
Q=(max(mean(N))−min(mean(N))mean(M))
(18)
5.11 Normalized Correlation (NK)
The similarity between two images is measured by this technique. A stronger positive correlation between the image (O) and image (N) is seen when value of NK is near to + 1 [53]. The Normalized Correlation equation is represented as:

NK=∑Pr=1∑Qs=1[O(r,s)×N(r,s)]∑Pr=1∑Qs=1[O(r,s)]2
(19)
5.12 Average Difference (AD)
The average difference between the image (O) and the image (N) is measured by this technique. The lower value of AD indicates more noise reduction in the denoised image [57]. The Average Difference equation is represented as:

AD=∑Pr=1∑Qs=1[O(r,s)−N(r,s)]P×Q
(20)
5.13 Normalized Absolute Error (NAE)
It is a measure of how far is the image (N) is from the image (O). A smaller value of NAE indicates good quality image [53]. The Normalized Absolute Error equation is represented as:

NAE=∑Pr=1∑Qs=1|O(r,s)×N(r,s)|∑Pr=1∑Qs=1|O(r,s)|
(21)
5.14 Structural Similarity Index Metric (SSIM)
This technique measures the similarity between the image (O) and image (N). The range of values for SSIM is [− 1, 1], where 1 is preferred to be the good similarity indicating the image (N) is of good quality and − 1 indicate bad similarity between image (O) and image (N). The Structural Similarity Index Metric equation is represented as:

SSIM=((2μpμq+C1)(2σpq+C2))((μ2p+μ2q+C1)(σ2p+σ2q+C2))
(22)
where μp
 is the average of (O), μq
 is average of (N) of common size P × P, σp,σq
 are standard deviation of original and denoised image, respectively. To remove the measurement inaccuracy two positive constant C1, C2 are used [57].

6 Results and Discussion
Noise-free synthetic images have been used for evaluation, which are then converted to noisy images by adding artificial speckle noise of variance 0.1. The size of all the images used is kept as 512 × 512. For performance evaluation, various parameters such as PSNR, MSE, RMSE, UQI, SNR, MAE, FSIM, SSI, MPSSI, SMPI, NK, AD, NAE and SSIM are evaluated. For evaluating the efficiency and accuracy of the proposed method, comparison is done with other methods.

In the performance analysis, the proposed method is compared with well-known methods which are selected based on their similarity to the proposed method. SRAD as proposed in [17] is based on anisotropic diffusion which preserves the image quality and removes speckle noise simply and quickly. Fourth Order Partial Differential Equations (FOPDE) as proposed in [58] minimizes a function which is proportional to the Laplacian absolute value of the image intensity function and works on piecewise planar images. Modified total variation (MTV) as proposed in [59] is a second order partial differential equation that minimizes the variation of image along the tangential direction to the original image isophotes. Perona Malik (PM) as proposed in [16] involves local and indentical computations over the entire image matrix which is simple. Split Bregman (SB) as proposed in [60] uses Bregman iteration to solve regularized optimization problems. Chambolle as proposed in [44] minimizes the total variation of an image which uses dual variables for optimization.

The parameter’s value used for the methods are mentioned as:

1.
SRAD with 200 iterations and 0.25 time-step [17].

2.
FOPDE with 500 iterations and 0.2 time-step [58].

3.
MTV with 200 iterations and 0.2 time-step [59].

4.
PM with 15 iterations [16].

5.
SB with 20 as regularization parameter [60].

6.
Chambolle with 0.83 as regularization parameter and 0.249 as tau used for calculating dual variables [44].

6.1 Performance Evaluation Using Synthetic Fetus Image
For the fetus synthetic image, Fig. 2a representing the original image and Fig. 2b representing the noisy image. Table 1 shows the values of parameters for various evaluation techniques. It can be observed from Fig. 2c and d that SRAD and FOPDE method exhibit speckle reduction, but output appear to be over-smoothened. The denoised image by MTV and SB method in Fig. 2e and g shows blurring of the necessary details with low PSNR values of 19.84 and 19.86 dB respectively. It can be observed from Fig. 2f that the PM method shows a clear denoised image and speckle noise is suppressed upto a certain amount. The denoised image from the Chambole method in Fig. 2h has some speckle residue, however the image is less blurry. The visual analysis shows that denoised image for the PM and proposed method are good, but for PM the denoised image still shows some amount of speckle. From Fig. 2i it can be seen that the proposed method has performed well both in terms of speckle noise suppression and maintaining the quality of image with better parameter values as seen from Table 1. The amount of smoothing is also balanced by the proposed method.

Fig. 2
figure 2
a Fetus synthetic image, b Noisy image; denoising by c SRAD d FOPDE e MTV f PM g SB h Chambolle i proposed method

Full size image
Table 1 Comparison of evaluation parameters for fetus synthetic image
Full size table
6.2 Performance Evaluation Using Synthetic Kidney Image
For the kidney synthetic image, we have Fig. 3a representing the original image. The denoised image by SRAD and MTV shown in Fig. 3c and e are blurred such that the sharp features are not much visible also integrating the low values of UQI, FSIM and SSIM from Table 2 for these methods it is inferred that the image quality is not very good. For the denoised image by the PM in Fig. 3f the image is over filtered due to which the details in the image are lost, this method has deteriorated the image structure. For FOPDE and SB the denoised image in Fig. 3d and g shows low contrast. For FOPDE the image details are not very clear. Figure 3h shows denoised image by the Chambolle method in which image detail are preserved also contrast is enhanced, however, speckle suppression is less due to which SMPI value in Table 2 is high. The denoised image by the proposed method in Fig. 3i shows a clean image with higher speckle noise reduction and the details in the image are well preserved. The UQI, FSIM and SSIM values are better than other methods for the proposed method.

Fig. 3
figure 3
a Kidney synthetic image, b Noisy image; denoising by c SRAD d FOPDE e MTV f PM g SB h Chambolle i Proposed Method

Full size image
Table 2 Comparison of evaluation parameters for kidney synthetic image
Full size table
6.3 Performance Evaluation Using the Phantom Image
For the phantom image, Fig. 4a represents the original image. The denoised image by SRAD and FOPDE shown in Fig. 4c and d in which the details are blurred. Figure 4e and h show denoised image by MTV and Chambolle, speckle suppression in MTV is more than Chambolle as seen from SMPI value in Table 3. However, for both MTV and Chambolle method SSIM value is less than the proposed method. The denoised image by the PM in Fig. 4f and SB in Fig. 4g both show clean image, but for PM there is distortion in the denoised image. For denoised image with SB display some speckle noise residue left in the image due to this the MPSSI value in Table 3 is more for SB than the PM. The denoised image in Fig. 4i by the proposed method shows balance between smoothing of image and speckle suppression. The Table 3 displays that all parameters of the proposed method are better than the other methods.

Fig. 4
figure 4
a Phantom image, b Noisy image; denoising by c SRAD d FOPDE e MTV f PM g SB h Chambolle i Proposed Method

Full size image
Table 3 Comparison of evaluation parameters for phantom image
Full size table
6.4 Performance Evaluation Using the Shepp-Logan Synthetic Image
For the shepp-logan synthetic image which is obtained from Matlab, Fig. 5a represents the original image. Denoised image by SRAD and SB methods shown in Fig. 5c and g respectively, clearly shows over-smoothing, the SSIM values in Table 4 for these two methods are thus lower than other methods. In case of denoised image of Fig. 5d and h, the image is clear, however, a significant amount of noise is still present, because of the presence of speckle noise MPSSI value is high for FOPDE and Chambolle method as seen in Table 4. The denoised image in Fig. 5e by MTV has speckle suppressed, but the image is blurred to a certain extent. Figure 5f shows the denoised image by the PM method which shows distortion in the white ring area of the image, however the speckle noise is removed from the image. The image denoised by the proposed method shown in Fig. 5i has effectively removed the speckle noise and has balanced the smoothing. The results from Table 4 shows that the values of the PM and proposed method are close, but the proposed method suppresses speckle noise very well.

Fig. 5
figure 5
a Shepp-logan synthetic image, b Noisy image; denoising by c SRAD d FOPDE e MTV f PM g SB h Chambolle i Proposed Method

Full size image
Table 4 Comparison of evaluation parameters for the Shepp-logan synthetic image
Full size table
6.5 Performance Evaluation Using Real US Image of the Carotid Artery
During the acquisition of real US images, the images contain speckle noise, thus it is difficult to obtain speckle-free original image, hence the quantitative evaluation is difficult. Thus, non-reference metrics have been used to evaluate the performance of methods for real US images in this paper along with visual analysis as shown in Fig. 6, Table 5 and Fig. 9, Table 8.

6.6 Performance Evaluation Using Real US Image of the Kidney
The performance evaluation is done as shown in Fig. 7, Table 6 and Fig. 8, Table 7  shows the parameters calculated for non-reference techniques SSI, MPSSI and SMPI, as these do not require noise-free original image. The visual analysis has also been considered because the absence of speckle-free original image makes the quantitative evaluation difficult. The reason for this is that for quantitative evaluation, it is mandatory for the original image to be speckle-free otherwise the parameters calculated are not accurate and during the acquisition of the real US image there is always the presence of speckle noise, thus the speckle-free image is hard to obtain. Figures 6, 7, 8 and 9 show the dancing results in real US images of kidney image and carotid artery. The value of the parameter is better for the proposed method. The visual analysis shows that the denoised image for the proposed method is better than other methods as the important structural details are preserved and the edges are preserved. The regularization done is sufficient so that the blurring effect is not seen, unlike in a denoised image by other methods like SRAD, MTV and SB. The Proposed method has maintained the contrast and enhanced the texture and organ surfaces. There is a good balance between speckle reduction and smoothing in the denoised images by the proposed method, also it has preserved the sharp features and anatomical structures in the image.

Table 5 Comparison of SSI, MPSSI and SMPI parameters for carotid artery image
Full size table
Table 6 Comparison of SSI, MPSSI and SMPI parameters for kidney image
Full size table
Table 7 Comparison of SSI, MPSSI and SMPI parameters for kidney image
Full size table
Table 8 Comparison of SSI, MPSSI and SMPI parameters for carotid artery image
Full size table
Fig. 6
figure 6
a Real US image of the carotid artery; denoising by b SRAD c FOPDE d MTV e PM f SB g Chambolle h Proposed Method

Full size image
Fig. 7
figure 7
a Real US image of kidney; denoising by b SRAD c FOPDE d MTV e PM f SB g Chambolle h Proposed Method

Full size image
Fig. 8
figure 8
a Real US image of kidney; denoising by b SRAD c FOPDE d MTV e PM f SB g Chambolle h Proposed Method

Full size image
Fig. 9
figure 9
a Real US image of Carotid Artery; denoising by b SRAD c FOPDE d MTV e PM f SB g Chambolle h Proposed Method

Full size image
7 Conclusion
The proposed method uses a combination of wavelets, total variation and morphological operations. The results obtained show that the proposed method is able to reduce speckle noise in an efficient manner as well as it performs well in preserving the important structural details in the US images. The evaluation is done on the basis of parameters PSNR, MSE, RMSE, UQI, SNR, MAE, FSIM, SSI, MPSSI, SMPI, NK, AD, NAE and SSIM which indicate good performance by the proposed method. The amount of regularization done by the proposed method is accurate as it does not involve over blurring of the denoised image. The image quality of the image denoised by the proposed method in case of real ultrasound images is better than other methods. The features in real ultrasound images need to be preserved as these contain information, but the removal of speckle noise may also remove these important features. In this paper, the combination used in the proposed method has the benefit as the wavelet transform lead to enhanced texture and organ surfaces and the use of optimization methods for total variation has reduced the number of iterations. The morphological operations have the benefit of edge preservation while retaining the structure of the denoised image. The median filter maintains the contrast level of the denoised image. The proposed method ensures to preserve structural information in US images after denoising. With structural information preserved, the visual perception is improved and US images can be examined by experts in a better way. US images of liver and other organs with the addition of higher amount of speckle noise can be used for denoising.

Abbreviations
US:
Ultrasound

TV:
Total variation

SRAD:
Speckle reducing anisotropic diffusion

PM:
Perona Malik

FOSRAD:
Faster oriented speckle reducing anisotropic diffusion

PSNR:
Peak signal to noise ratio

MSE:
Mean squared error

RMSE:
Root mean squared error

UQI:
Universal Quality Index

SNR:
Signal to noise ratio

MAE:
Mean absolute error

FSIM:
Feature Similarity Index Metric

SSI:
Speckle Suppression Index

MPSSI:
Mean Preservation Speckle Suppression Index

SMPI:
Speckle Suppression and Mean Preservation Index

NK:
Normalized correlation

AD:
Average difference

NAE:
Normalized absolute error

SSIM:
Structural Similarity Index Metric

FOPDE:
Fourth order partial differential equations

MTV:
Modified total variation

SB:
Split Bregman

References
