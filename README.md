# ac
actor critic algorithm


## intro
**Basic Actor Critic Algorithm**
```python
#gradient policy, to max goal(cumulative rewards)
actor_loss = -(pi.log_prob(a) * td.detach() + 0.005 * entropy)
#logical value should close to real value (reward)
td = v_targets - values # logical reality difference
c_loss = td.pow(2)
```