import sys
sys.path.insert(0, '.')
from train import get_action
from app.environment import EnterpriseAPEnvironment
from app.models import Action

env = EnterpriseAPEnvironment('expert_negotiation')
obs = env.reset().model_dump()
step = 0
done = False
while not done and step < 30:
    a = get_action('expert_negotiation', obs, step, 60, 60)
    print(f"Step {step} action: {a}")
    res = env.step(Action(**a)).model_dump()
    obs = res['observation']
    done = res['done']
    step += 1

print(f"Done! Final extracted fields: {obs.get('extracted_fields')}")
print(f"Final score: {res.get('info', {}).get('final_score')}")
