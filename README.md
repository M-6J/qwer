# SNN_MOBILE



#### ( Object Detection with Spiking Neural Networks on Automotive Event Data )github

> https://github.com/loiccordone/object-detection-with-spiking-neural-networks

빠르고 효율적인 자동차 내장 app설계하기위해 event cameras에 나오는 데이터에 직접 snn을 훈련시킴

 

spike backpropagation – surrogate(대리) gradient learning, parametric LIF, SpikingJelly framework and of our new voxel(3D화소) cube event encoding를 사용해 SNN을 기반으로 SqueezeNet, VGG, MobileNet, DenseNet 를 학습

two automotive event datasets에 대한 실험, SNN을 SSD(SingleShot MultiBox Detector) 와 결합해 복잡한 GEN1 automotive detection event dataset에서 object detection을 할 수 있는 snn 제안.

 

SSD bounding box regression heads to design the first spiking neural networks capable of doing object detection on the real-world event dataset

SSD 경계 상자 회귀 헤드는 실제 이벤트 데이터 세트(Prophesee GEN1)에서 객체 감지를 수행할 수 있는 최초의 스파이킹 신경망을 설계합니다.

![spiking_DenseNet+SSD_architecture](C:\Users\jmk07\Desktop\snn_mobile\Readme_image\spiking_DenseNet+SSD_architecture.png)

오랫동안 small dataset으로 제한되었던 snn은 이제 temporal data에 대해 직접 훈련될 때 장점이있다