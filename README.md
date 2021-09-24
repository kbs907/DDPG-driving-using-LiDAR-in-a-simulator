# DDPG driving using LiDAR in a simulator
 
## Video
[![Watch the video](https://img.youtube.com/vi/N2MVnuQA8GQ/maxresdefault.jpg)](https://youtu.be/N2MVnuQA8GQ)

## Goal

기존에 자이트론에서 제공해준 초음파 센서를 이용한 DQN 주행 코드에서
1. 라이다 센서를 활용
2. DDPG 학습 모델 추가 적용

</br>

## Environment

* Ubuntu 18.04
* base code using DQN in simulator

</br>

## Key Code

자이트론에서 제공해준 DQN 학습 코드에 융합 적용했기에 라이센스 문제로 독립적으로 보여줄 수 있는 일부 코드만 공개

~~~
class Actor(nn.Module):	#@@@#
    def __init__(self, state_dim, action_dim, max_action, hidden_size):
        super(Actor, self).__init__()
		
        modullist = []
        modullist.append(("InputLayer", nn.Linear(state_dim, hidden_size[0])))
        cnt = 0
        for layer in range(len(hidden_size)-1):
            modullist.append(("Relu_"+str(cnt), nn.ReLU()))
            modullist.append(("hiddenlayer_"+str(cnt), nn.Linear(hidden_size[layer], hidden_size[layer+1])))
            cnt += 1
  
        modullist.append(("Relu_"+str(cnt), nn.ReLU()))
        modullist.append(("OutputLayer", nn.Linear(hidden_size[len(hidden_size)-1], action_dim)))
		
        self.actor_model = nn.Sequential(OrderedDict(modullist))
        print(self.actor_model)

        self.max_action = max_action

    def forward(self, x):
        x = self.actor_model(x)
        x = self.max_action * torch.tanh(x)
        return x
	
    def sample_action(self, obs):
        out = self.forward(obs)
        return out.item()		#!?# or return out

class OUNoise():
    def __init__(self, OU_param):
        self.X = np.zeros(1)
        self.mu = OU_param["mu"]
        self.theta = OU_param["theta"]
        self.sigma = OU_param["sigma"]

    def sample(self):
        dx = self.theta * (self.mu - self.X) + self.sigma * np.random.randn(len(self.X))
        self.X += dx
        return torch.from_numpy(self.X).float()

class Critic(nn.Module):	#@@@#
    def __init__(self, state_dim, action_dim, hidden_size):
        super(Critic, self).__init__()

        modullist = []
        modullist.append(("InputLayer", nn.Linear(state_dim+action_dim, hidden_size[0])))
        cnt = 0
        for layer in range(len(hidden_size)-1):
            modullist.append(("Relu_"+str(cnt), nn.ReLU()))
            modullist.append(("hiddenlayer_"+str(cnt), nn.Linear(hidden_size[layer], hidden_size[layer+1])))
            cnt += 1
  
        modullist.append(("Relu_"+str(cnt), nn.ReLU()))
        modullist.append(("OutputLayer", nn.Linear(hidden_size[len(hidden_size)-1], 1)))
		
        self.critic_model = nn.Sequential(OrderedDict(modullist))
        print(self.critic_model)

    def forward(self, x, u):
        x = self.critic_model(torch.cat([x,u], 1))
        return x
~~~
</br>

## Limitations

* DDPG 적용이 목표였기에, 학습이 되는지만 확인한 정도
* 더 나은 학습 성능을 보이기 위해 파라미터 조정, 보상 설계 또는 추가적인 기법 적용이 필요


</br>

## What I've learned

* 새로 적용하고자 하는 것과 기존 틀을 이해하고, 기존 틀에 맞춰 융합하는 방법
* DQN, DDPG 모델에 대한 이해
* pytorch에 대한 이해

</br>

## What I have to do in the future

1. limitation 극복
2. camera와 CNN을 활용한 DDPG 주행 모델 적용 및 확인
