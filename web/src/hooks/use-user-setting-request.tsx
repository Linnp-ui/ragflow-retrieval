import message from '@/components/ui/message';
import { ResponseGetType } from '@/interfaces/database/base';
import { IToken } from '@/interfaces/database/chat';
import { ITenantInfo } from '@/interfaces/database/dataset';
import { ILangfuseConfig } from '@/interfaces/database/system';
import {
  ITenant,
  ITenantUser,
  IUserInfo,
} from '@/interfaces/database/user-setting';
import { ISetLangfuseConfigRequestBody } from '@/interfaces/request/system';
import { DEFAULT_LANGUAGE_CODE, supportedLanguages } from '@/locales/config';
import userService, {
  addTenantUser,
  agreeTenant,
  deleteTenantUser,
  listTenant,
  listTenantUser,
} from '@/services/user-service';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { useWarnEmptyModel } from './use-warn-empty-model';

export const enum UserSettingApiAction {
  UserInfo = 'userInfo',
  TenantInfo = 'tenantInfo',
  SaveSetting = 'saveSetting',
  FetchManualSystemTokenList = 'fetchManualSystemTokenList',
  FetchSystemTokenList = 'fetchSystemTokenList',
  RemoveSystemToken = 'removeSystemToken',
  CreateSystemToken = 'createSystemToken',
  ListTenantUser = 'listTenantUser',
  AddTenantUser = 'addTenantUser',
  DeleteTenantUser = 'deleteTenantUser',
  ListTenant = 'listTenant',
  AgreeTenant = 'agreeTenant',
  SetLangfuseConfig = 'setLangfuseConfig',
  DeleteLangfuseConfig = 'deleteLangfuseConfig',
  FetchLangfuseConfig = 'fetchLangfuseConfig',
}

export const useFetchUserInfo = (): ResponseGetType<IUserInfo> => {
  const { data, isFetching: loading } = useQuery({
    queryKey: [UserSettingApiAction.UserInfo],
    initialData: {},
    gcTime: 0,
    queryFn: async () => {
      const { data } = await userService.userInfo();

      if (data.code === 0) {
        const targetLng =
          supportedLanguages.find((lang) => lang.code === data.data.language)
            ?.code ?? DEFAULT_LANGUAGE_CODE;

        return Object.assign({}, data.data, {
          language: targetLng,
        });
      }

      return data.data ?? {};
    },
  });

  return { data, loading };
};

// Stop using this interface to retrieve the default model; instead, directly call `useFetchDefaultModelDictionary`.
export const useFetchTenantInfo = (
  showEmptyModelWarn = false,
): ResponseGetType<ITenantInfo> => {
  const { data, isFetching: loading } = useQuery({
    queryKey: [UserSettingApiAction.TenantInfo, showEmptyModelWarn],
    initialData: {},
    gcTime: 0,
    queryFn: async () => {
      const { data: res } = await userService.getTenantInfo();
      if (res.code === 0) {
        // llm_id is chat_id
        // asr_id is speech2txt
        const { data } = res;
        data.chat_id = data.llm_id;
        data.speech2text_id = data.asr_id;

        return data;
      }

      return res;
    },
  });

  useWarnEmptyModel(showEmptyModelWarn, data?.embd_id, data?.llm_id, loading);

  return { data, loading };
};

export const useSelectParserList = (): Array<{ value: string; label: string; description?: string; }> => {
  const { data: tenantInfo } = useFetchTenantInfo(true);
  const { t } = useTranslation();

  const defaultParsers = useMemo(
    () => [
      { value: 'naive', label: t('knowledgeConfiguration.parserLabel.naive'), description: t('knowledgeConfiguration.parserDescription.naive') },
      { value: 'qa', label: t('knowledgeConfiguration.parserLabel.qa'), description: t('knowledgeConfiguration.parserDescription.qa') },
      {
        value: 'resume',
        label: t('knowledgeConfiguration.parserLabel.resume'),
        description: t('knowledgeConfiguration.parserDescription.resume'),
      },
      {
        value: 'manual',
        label: t('knowledgeConfiguration.parserLabel.manual'),
        description: t('knowledgeConfiguration.parserDescription.manual'),
      },
      { value: 'table', label: t('knowledgeConfiguration.parserLabel.table'), description: t('knowledgeConfiguration.parserDescription.table') },
      { value: 'paper', label: t('knowledgeConfiguration.parserLabel.paper'), description: t('knowledgeConfiguration.parserDescription.paper') },
      { value: 'book', label: t('knowledgeConfiguration.parserLabel.book'), description: t('knowledgeConfiguration.parserDescription.book') },
      { value: 'laws', label: t('knowledgeConfiguration.parserLabel.laws'), description: t('knowledgeConfiguration.parserDescription.laws') },
      {
        value: 'presentation',
        label: t('knowledgeConfiguration.parserLabel.presentation'),
        description: t('knowledgeConfiguration.parserDescription.presentation'),
      },
      {
        value: 'picture',
        label: t('knowledgeConfiguration.parserLabel.picture'),
        description: t('knowledgeConfiguration.parserDescription.picture'),
      },
      { value: 'one', label: t('knowledgeConfiguration.parserLabel.one'), description: t('knowledgeConfiguration.parserDescription.one') },
      { value: 'audio', label: t('knowledgeConfiguration.parserLabel.audio'), description: t('knowledgeConfiguration.parserDescription.audio') },
      { value: 'email', label: t('knowledgeConfiguration.parserLabel.email'), description: t('knowledgeConfiguration.parserDescription.email') },
      { value: 'tag', label: t('knowledgeConfiguration.parserLabel.tag'), description: t('knowledgeConfiguration.parserDescription.tag') },
    ],
    [t],
  );

  const parserList = useMemo(() => {
    const parserArray: Array<string> = tenantInfo?.parser_ids?.split(',') ?? [];
    const filteredArray = parserArray.filter((x) => x.trim() !== '');

    if (filteredArray.length === 0) {
      return defaultParsers;
    }

    return filteredArray.map((x) => {
      const arr = x.split(':');
      return { value: arr[0], label: arr[1] };
    });
  }, [tenantInfo, defaultParsers]);

  return parserList;
};

export const useSaveSetting = (silent = false) => {
  const queryClient = useQueryClient();
  const { t } = useTranslation();
  const {
    data,
    isPending: loading,
    mutateAsync,
  } = useMutation({
    mutationKey: [UserSettingApiAction.SaveSetting],
    mutationFn: async (
      userInfo: { new_password: string } | Partial<IUserInfo>,
    ) => {
      const { data } = await userService.setting(userInfo);
      if (data.code === 0) {
        if (!silent) {
          message.success(t('message.modified'));
        }
        queryClient.invalidateQueries({ queryKey: ['userInfo'] });
      }
      return data?.code;
    },
  });

  return { data, loading, saveSetting: mutateAsync };
};

export const useFetchSystemVersion = () => {
  const [version, setVersion] = useState('');
  const [loading, setLoading] = useState(false);

  const fetchSystemVersion = useCallback(async () => {
    try {
      setLoading(true);
      const { data } = await userService.getSystemVersion();
      if (data.code === 0) {
        setVersion(data.data);
        setLoading(false);
      }
    } catch (error) {
      console.warn(error);
      setLoading(false);
    }
  }, []);

  return { fetchSystemVersion, version, loading };
};

export const useFetchManualSystemTokenList = () => {
  const {
    data,
    isPending: loading,
    mutateAsync,
  } = useMutation({
    mutationKey: [UserSettingApiAction.FetchManualSystemTokenList],
    mutationFn: async () => {
      const { data } = await userService.listToken();

      return data?.data ?? [];
    },
  });

  return { data, loading, fetchSystemTokenList: mutateAsync };
};

export const useFetchSystemTokenList = () => {
  const {
    data,
    isFetching: loading,
    refetch,
  } = useQuery<IToken[]>({
    queryKey: [UserSettingApiAction.FetchSystemTokenList],
    initialData: [],
    gcTime: 0,
    queryFn: async () => {
      const { data } = await userService.listToken();

      return data?.data ?? [];
    },
  });

  return { data, loading, refetch };
};

export const useRemoveSystemToken = () => {
  const queryClient = useQueryClient();
  const { t } = useTranslation();

  const {
    data,
    isPending: loading,
    mutateAsync,
  } = useMutation({
    mutationKey: [UserSettingApiAction.RemoveSystemToken],
    mutationFn: async (token: string) => {
      const { data } = await userService.removeToken({}, token);
      if (data.code === 0) {
        message.success(t('message.deleted'));
        queryClient.invalidateQueries({
          queryKey: [UserSettingApiAction.FetchSystemTokenList],
        });
      }
      return data?.data ?? [];
    },
  });

  return { data, loading, removeToken: mutateAsync };
};

export const useCreateSystemToken = () => {
  const queryClient = useQueryClient();

  const {
    data,
    isPending: loading,
    mutateAsync,
  } = useMutation({
    mutationKey: [UserSettingApiAction.CreateSystemToken],
    mutationFn: async (params: Record<string, any>) => {
      const { data } = await userService.createToken(params);
      if (data.code === 0) {
        queryClient.invalidateQueries({
          queryKey: [UserSettingApiAction.FetchSystemTokenList],
        });
      }
      return data?.data ?? [];
    },
  });

  return { data, loading, createToken: mutateAsync };
};

export const useListTenantUser = () => {
  const { data: tenantInfo } = useFetchTenantInfo();
  const tenantId = tenantInfo.tenant_id;
  const {
    data,
    isFetching: loading,
    refetch,
  } = useQuery<ITenantUser[]>({
    queryKey: [UserSettingApiAction.ListTenantUser, tenantId],
    initialData: [],
    gcTime: 0,
    enabled: !!tenantId,
    queryFn: async () => {
      const { data } = await listTenantUser(tenantId);

      return data?.data ?? [];
    },
  });

  return { data, loading, refetch };
};

export const useAddTenantUser = () => {
  const { data: tenantInfo } = useFetchTenantInfo();
  const queryClient = useQueryClient();
  const {
    data,
    isPending: loading,
    mutateAsync,
  } = useMutation({
    mutationKey: [UserSettingApiAction.AddTenantUser],
    mutationFn: async (email: string) => {
      const { data } = await addTenantUser(tenantInfo.tenant_id, email);
      if (data.code === 0) {
        queryClient.invalidateQueries({
          queryKey: [UserSettingApiAction.ListTenantUser],
        });
      }
      return data?.code;
    },
  });

  return { data, loading, addTenantUser: mutateAsync };
};

export const useDeleteTenantUser = () => {
  const { data: tenantInfo } = useFetchTenantInfo();
  const queryClient = useQueryClient();
  const { t } = useTranslation();

  const {
    data,
    isPending: loading,
    mutateAsync,
  } = useMutation({
    mutationKey: [UserSettingApiAction.DeleteTenantUser],
    mutationFn: async ({
      userId,
      tenantId,
    }: {
      userId: string;
      tenantId?: string;
    }) => {
      const { data } = await deleteTenantUser({
        tenantId: tenantId ?? tenantInfo.tenant_id,
        userId,
      });
      if (data.code === 0) {
        message.success(t('message.deleted'));
        queryClient.invalidateQueries({
          queryKey: [UserSettingApiAction.ListTenantUser],
        });
        queryClient.invalidateQueries({
          queryKey: [UserSettingApiAction.ListTenant],
        });
      }
      return data?.data ?? [];
    },
  });

  return { data, loading, deleteTenantUser: mutateAsync };
};

export const useListTenant = () => {
  const { data: tenantInfo } = useFetchTenantInfo();
  const tenantId = tenantInfo.tenant_id;
  const {
    data,
    isFetching: loading,
    refetch,
  } = useQuery<ITenant[]>({
    queryKey: [UserSettingApiAction.ListTenant, tenantId],
    initialData: [],
    gcTime: 0,
    enabled: !!tenantId,
    queryFn: async () => {
      const { data } = await listTenant();

      return data?.data ?? [];
    },
  });

  return { data, loading, refetch };
};

export const useAgreeTenant = () => {
  const queryClient = useQueryClient();
  const { t } = useTranslation();

  const {
    data,
    isPending: loading,
    mutateAsync,
  } = useMutation({
    mutationKey: [UserSettingApiAction.AgreeTenant],
    mutationFn: async (tenantId: string) => {
      const { data } = await agreeTenant(tenantId);
      if (data.code === 0) {
        message.success(t('message.operated'));
        queryClient.invalidateQueries({
          queryKey: [UserSettingApiAction.ListTenant],
        });
      }
      return data?.data ?? [];
    },
  });

  return { data, loading, agreeTenant: mutateAsync };
};

export const useSetLangfuseConfig = () => {
  const { t } = useTranslation();
  const {
    data,
    isPending: loading,
    mutateAsync,
  } = useMutation({
    mutationKey: [UserSettingApiAction.SetLangfuseConfig],
    mutationFn: async (params: ISetLangfuseConfigRequestBody) => {
      const { data } = await userService.setLangfuseConfig(params);
      if (data.code === 0) {
        message.success(t('message.operated'));
      }
      return data?.code;
    },
  });

  return { data, loading, setLangfuseConfig: mutateAsync };
};

export const useDeleteLangfuseConfig = () => {
  const { t } = useTranslation();
  const {
    data,
    isPending: loading,
    mutateAsync,
  } = useMutation({
    mutationKey: [UserSettingApiAction.DeleteLangfuseConfig],
    mutationFn: async () => {
      const { data } = await userService.deleteLangfuseConfig();
      if (data.code === 0) {
        message.success(t('message.deleted'));
      }
      return data?.code;
    },
  });

  return { data, loading, deleteLangfuseConfig: mutateAsync };
};

export const useFetchLangfuseConfig = () => {
  const { data, isFetching: loading } = useQuery<ILangfuseConfig>({
    queryKey: [UserSettingApiAction.FetchLangfuseConfig],
    gcTime: 0,
    queryFn: async () => {
      const { data } = await userService.getLangfuseConfig();

      return data?.data;
    },
  });

  return { data, loading };
};
