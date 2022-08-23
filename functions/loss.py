import torch

#means = make all potential increment around the mean(default = 1.0)
#lamb(이상치) = adjust the norm factor to avoid outlier (default: 0.0)
def TET_loss(outputs, labels, criterion, means, lamb):
    #T = total simulation time
    T = outputs.size(1)
    Loss_es = 0
    for t in range(T): #T-1 번 반복
        # 손실을 계산하기 위해 네트워크의 출력을 레이블과 함께 전달, main.py에 93줄
        # criterion = nn.CrossEntropyLoss().to(device)
        Loss_es += criterion(outputs[:, t, ...], labels) #TET Eq 8 다시봐야함
    Loss_es = Loss_es / T # L_TET, TET Eq.9
    if lamb != 0:
        MMDLoss = torch.nn.MSELoss()
        y = torch.zeros_like(outputs).fill_(means)
        Loss_mmd = MMDLoss(outputs, y) # L_mse, TET Eq.12
    else:
        Loss_mmd = 0
    return (1 - lamb) * Loss_es + lamb * Loss_mmd # L_Total, TET Eq.13